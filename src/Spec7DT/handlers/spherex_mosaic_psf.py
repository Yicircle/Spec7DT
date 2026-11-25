"""Build stationary effective-PSF manifests for SPHEREx mosaic cubes.

All SPHEREx-specific exposure discovery, WCS-WAVE interpretation, detector
ePSF selection, and equal-hit reprojection live in this module.  The generic
safe-PSF provider consumes only the generated schema-v2 manifest.
"""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, replace
from hashlib import sha256
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import shutil
import tempfile
from threading import Event
from typing import Any, Literal, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.interpolate import RegularGridInterpolator

from ..psf.models import (
    ImageProductMetadata,
    PSFEnsemble,
    PSFSample,
)
from ..psf.products import sampling_from_header
from ..psf.render import (
    render_oversampled_psf,
    representative_from_ensemble,
    summarize_spatial_ensemble,
    validate_effective_psf,
)
from .spherex_psf import SpherexEPSFProvider, _version_major


_SPATIAL_R80_TOLERANCE = 0.02
_SPATIAL_EE_TOLERANCE = 0.01
_SPATIAL_L1_TOLERANCE = 0.02
_SPATIAL_CENTROID_TOLERANCE = 0.05
_PSF_EDGE_TOLERANCE = 1e-4
_NEGATIVE_PEAK_TOLERANCE = 1e-8
_NEGATIVE_MASS_TOLERANCE = 1e-6
_LEVEL2_STAMP_SIZE = 11
_IRSA_HOSTS = {
    "irsa.ipac.caltech.edu",
    "nasa-irsa-spherex.s3.us-east-1.amazonaws.com",
}
# IRSA flag bit numbers documented for mosaic overflow and outlier handling.
_DEFAULT_REJECT_FLAG_MASK = (1 << 1) | (1 << 19)


class SpherexMosaicPSFBuildError(RuntimeError):
    """Aggregated failure raised before a generated manifest becomes active."""

    def __init__(self, failures):
        self.failures = dict(failures)
        lines = ["SPHEREx mosaic PSF synthesis failed; no manifest was activated:"]
        lines.extend(f"  {key}: {value}" for key, value in sorted(self.failures.items()))
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class SpherexMosaicPSFSynthesisConfig:
    exposure_list: str | Path | Mapping | Sequence[Mapping] | None = None
    query_if_missing: bool = False
    data_collection: Literal["spherex_qr3", "spherex_qr2"] = "spherex_qr3"
    network_policy: Literal["offline", "cache"] = "offline"
    cache_dir: str | Path | None = None
    artifact_dir: str | Path | None = None
    tile_bounds: Mapping[str, tuple[float, float, float, float]] | None = None
    max_workers: int = 4
    progress: bool = True
    download_retries: int = 3
    retry_backoff_seconds: float = 1.0
    download_timeout_seconds: float = 120.0

    def __post_init__(self):
        if self.data_collection not in {"spherex_qr3", "spherex_qr2"}:
            raise ValueError("data_collection must be spherex_qr3 or spherex_qr2")
        if self.network_policy not in {"offline", "cache"}:
            raise ValueError("network_policy must be offline or cache")
        if not isinstance(self.max_workers, int) or self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        if not isinstance(self.progress, bool):
            raise TypeError("progress must be bool")
        if (
            not isinstance(self.download_retries, int)
            or isinstance(self.download_retries, bool)
            or self.download_retries < 0
        ):
            raise ValueError("download_retries must be a non-negative integer")
        if (
            not np.isfinite(self.retry_backoff_seconds)
            or self.retry_backoff_seconds < 0
        ):
            raise ValueError("retry_backoff_seconds must be finite and non-negative")
        if (
            not np.isfinite(self.download_timeout_seconds)
            or self.download_timeout_seconds <= 0
        ):
            raise ValueError("download_timeout_seconds must be finite and positive")
        if self.query_if_missing and self.network_policy != "cache":
            raise ValueError("query_if_missing=True requires network_policy='cache'")
        if self.tile_bounds is not None:
            normalized = {}
            for key, value in dict(self.tile_bounds).items():
                bounds = tuple(float(item) for item in value)
                if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
                    raise ValueError("tile_bounds values must be finite (x0, y0, x1, y1)")
                if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    raise ValueError("tile_bounds must have positive width and height")
                normalized[str(key)] = bounds
            object.__setattr__(self, "tile_bounds", normalized)

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("mosaic_synthesis must be a mapping or SpherexMosaicPSFSynthesisConfig")

    @property
    def resolved_cache_dir(self):
        if self.cache_dir is not None:
            return Path(self.cache_dir).expanduser().resolve()
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return (base / "spec7dt" / "spherex_psf").resolve()

    @property
    def resolved_artifact_dir(self):
        if self.artifact_dir is not None:
            return Path(self.artifact_dir).expanduser().resolve()
        return self.resolved_cache_dir / "generated"


@dataclass(frozen=True)
class SpherexMosaicPSFBuildResult:
    manifest: Mapping
    manifest_path: Path
    exposure_snapshot_path: Path
    artifact_paths: tuple[Path, ...]
    qc: Mapping


@dataclass
class _ExposureData:
    record: dict
    path: Path
    primary_header: fits.Header
    image_header: fits.Header
    image: np.ndarray
    flags: np.ndarray
    wcs: WCS
    sampling: Any
    wavelength: Any
    product: ImageProductMetadata
    epsf_table: np.ndarray
    epsf_header: fits.Header
    epsf_names: dict[str, str]


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(value):
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _clean_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    if np.ma.is_masked(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first(record, names):
    lower = {str(key).lower(): key for key in record}
    for name in names:
        key = lower.get(name.lower())
        if key is not None:
            value = _clean_scalar(record[key])
            if value is not None and str(value).strip():
                return value
    return None


def _normalize_exposure(record, *, base=None, index=0):
    item = {str(key): _clean_scalar(value) for key, value in dict(record).items()}
    path = _first(item, ("path", "filepath", "local_path"))
    access_url = _first(item, ("access_url", "url", "download_url"))
    if path is not None:
        path = Path(str(path)).expanduser()
        if not path.is_absolute() and base is not None:
            path = Path(base) / path
        path = str(path.resolve())
    if path is None and access_url is None:
        raise ValueError("each exposure requires path or access_url")
    exposure_id = _first(
        item,
        ("exposure_id", "artifact_id", "obs_publisher_did", "lvf_id", "obs_id"),
    )
    if exposure_id is None:
        raise ValueError(
            "each exposure requires exposure_id or an lvf_id/obs_id/artifact_id alias"
        )
    detector = _first(item, ("detector", "det_id"))
    result = {
        "exposure_id": str(exposure_id),
        "path": path,
        "access_url": None if access_url is None else str(access_url),
        "detector": None if detector is None else int(detector),
        "release": _first(
            item,
            ("release", "data_collection", "obs_collection", "collection"),
        ),
        "sha256": _first(item, ("sha256", "checksum")),
    }
    for name in ("s_region", "dataproduct_subtype"):
        value = _first(item, (name,))
        if value is not None:
            result[name] = value
    return result


def _table_exposure_list(path):
    table = Table.read(path)
    records = [
        {name: row[name] for name in table.colnames}
        for row in table
    ]
    return {
        "selection": str(table.meta.get("selection", "candidate")).lower(),
        "data_collection": table.meta.get("data_collection"),
        "exposures": records,
    }


def _select_exposure_group(raw, group_id):
    if isinstance(raw, Mapping) and "groups" in raw:
        matches = [
            item for item in raw["groups"]
            if str(item.get("product_group_id")) == str(group_id)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"exposure list has {len(matches)} entries for product_group_id {group_id}"
            )
        return dict(matches[0])
    if isinstance(raw, Mapping) and "exposures" in raw:
        declared = raw.get("product_group_id")
        if declared is not None and str(declared) != str(group_id):
            raise ValueError(
                f"exposure list targets product_group_id {declared}, not {group_id}"
            )
        return dict(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Path)):
        return {"selection": "exact", "exposures": list(raw)}
    raise TypeError("unsupported exposure-list structure")


def _load_explicit_exposure_list(value, group_id):
    if value is None:
        return None
    base = None
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"exposure list does not exist: {path}")
        base = path.parent
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            raw = _table_exposure_list(path)
    else:
        raw = value
    if isinstance(raw, Mapping) and int(raw.get("schema_version", 1)) != 1:
        raise ValueError("unsupported exposure-list schema_version")
    selected = _select_exposure_group(raw, group_id)
    selection = str(selected.get("selection", "exact")).lower()
    if selection not in {"exact", "candidate"}:
        raise ValueError("exposure-list selection must be exact or candidate")
    records = [
        _normalize_exposure(item, base=base, index=index)
        for index, item in enumerate(selected.get("exposures", ()))
    ]
    if not records:
        raise ValueError("exposure list is empty")
    ids = [item["exposure_id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("exposure list contains duplicate exposure_id values")
    return {
        "schema_version": 1,
        "kind": "spherex_level2_exposure_list",
        "product_group_id": str(group_id),
        "selection": selection,
        "data_collection": selected.get("data_collection"),
        "exposures": records,
    }


def _url_identity(value):
    """Return the immutable Level-2 object URL without cutout parameters."""

    parsed = urlparse(str(value))
    return urlunparse(parsed._replace(query="", fragment=""))


def _collection_from_url(value):
    parts = [part.lower() for part in urlparse(str(value)).path.split("/")]
    for release in ("qr3", "qr2"):
        if release in parts:
            return f"spherex_{release}"
    return None


def _load_mosaic_inputs(product, group_id, config):
    """Read an exact Level-2 list embedded by the SPHEREx mosaic maker."""

    if product.source_path is None:
        return None
    path = Path(product.source_path).expanduser().resolve()
    if not path.is_file():
        return None
    hdu_name = str(product.provenance.get("inputs_hdu") or "INPUTS")
    with fits.open(path, memmap=True, lazy_load_hdus=False) as hdul:
        if hdu_name not in hdul:
            return None
        table = hdul[hdu_name].data
        names = list(hdul[hdu_name].columns.names or ())
        if table is None or not len(table):
            raise ValueError(f"mosaic {hdu_name} HDU is empty: {path}")
        required = {"OBS_ID", "DETECTOR"}
        missing = sorted(required - {str(name).upper() for name in names})
        if missing:
            raise ValueError(
                f"mosaic {hdu_name} HDU is missing columns: {', '.join(missing)}"
            )
        raw_rows = [
            {name: _clean_scalar(row[name]) for name in names}
            for row in table
        ]

    records = []
    collections = set()
    seen_ids = set()
    seen_urls = set()
    for index, row in enumerate(raw_rows):
        detector = int(_first(row, ("detector", "det_id")))
        obs_id = str(_first(row, ("obs_id", "exposure_id", "lvf_id")))
        exposure_id = f"{obs_id}:D{detector}"
        if exposure_id in seen_ids:
            raise ValueError(
                f"mosaic {hdu_name} contains duplicate exposure {exposure_id}"
            )
        seen_ids.add(exposure_id)

        local_path = _first(row, ("local_path", "path", "filepath"))
        if local_path is not None:
            local_path = Path(str(local_path)).expanduser()
            if not local_path.is_file():
                local_path = None
        access_url = _first(row, ("access_url", "url", "download_url"))
        if access_url is not None:
            access_url = _url_identity(access_url)
            if access_url in seen_urls:
                raise ValueError(
                    f"mosaic {hdu_name} contains duplicate ACCESS_URL: {access_url}"
                )
            seen_urls.add(access_url)
            collection = _collection_from_url(access_url)
            if collection is not None:
                collections.add(collection)
        if local_path is None and access_url is None:
            raise ValueError(
                f"mosaic {hdu_name} row {index} has neither a usable LOCAL_PATH "
                "nor ACCESS_URL"
            )
        record = _normalize_exposure({
            "exposure_id": exposure_id,
            "path": None if local_path is None else str(local_path.resolve()),
            "access_url": access_url,
            "detector": detector,
            "release": next(iter(collections)) if len(collections) == 1 else None,
            "sha256": _first(row, ("sha256", "checksum")),
        }, index=index)
        records.append(record)

    if len(collections) > 1:
        raise ValueError(
            f"mosaic {hdu_name} mixes data collections: {sorted(collections)}"
        )
    embedded_collection = next(iter(collections), None)
    if (
        embedded_collection is not None
        and embedded_collection != config.data_collection
    ):
        raise ValueError(
            f"mosaic {hdu_name} uses {embedded_collection}, but mosaic synthesis "
            f"is configured for {config.data_collection}"
        )
    for record in records:
        record["release"] = embedded_collection or config.data_collection
    return {
        "schema_version": 1,
        "kind": "spherex_level2_exposure_list",
        "product_group_id": str(group_id),
        "selection": "exact",
        "data_collection": embedded_collection or config.data_collection,
        "source": "mosaic_inputs",
        "source_path": str(path),
        "source_hdu": hdu_name,
        "exposures": records,
    }


def _footprint(header, shape, bounds):
    wcs = WCS(header, relax=True).celestial
    points = np.asarray(_sample_positions(bounds), dtype=float)
    sky = wcs.pixel_to_world(points[:, 0], points[:, 1])
    center = sky[0]
    radius = float(np.max(center.separation(sky[1:]).degree))
    return center, max(radius, 1.0 / 3600.0), sky


def _query_irsa_exposures(sample_skies, collection):
    from astroquery.ipac.irsa import Irsa

    by_id = {}
    for sample_index, sky in enumerate(sample_skies):
        table = Irsa.query_sia(
            pos=(sky, 1.0 * u.arcsec),
            collection=str(collection),
            calib_level=2,
            maxrec=100000,
        )
        for index, row in enumerate(table):
            item = {name: row[name] for name in table.colnames}
            normalized = _normalize_exposure(item, index=index)
            exposure_id = normalized["exposure_id"]
            previous = by_id.get(exposure_id)
            if previous is None:
                normalized["_query_sample_indices"] = [sample_index]
                by_id[exposure_id] = normalized
                continue
            for key in ("access_url", "path", "detector", "release"):
                if (
                    previous.get(key) is not None
                    and normalized.get(key) is not None
                    and str(previous[key]) != str(normalized[key])
                ):
                    raise ValueError(
                        f"IRSA SIA2 changed {key} for exposure {exposure_id}"
                    )
            previous["_query_sample_indices"].append(sample_index)
    records = list(by_id.values())
    if not records:
        raise ValueError("IRSA SIA2 returned no Level-2 exposures for the mosaic footprint")
    return sorted(records, key=lambda item: item["exposure_id"])


def _coverage_cache_path(records, sample_skies, collection, config):
    payload = {
        "collection": str(collection),
        "exposures": sorted(
            _url_identity(item["access_url"])
            for item in records
            if item.get("access_url") is not None
        ),
        "samples_deg": [
            [round(float(sky.ra.degree), 10), round(float(sky.dec.degree), 10)]
            for sky in sample_skies
        ],
    }
    return (
        config.resolved_cache_dir
        / "discovery"
        / f"{_stable_digest(payload)}.json"
    )


def _assign_embedded_input_samples(snapshot, products, sample_skies, config):
    """Intersect exact INPUTS rows with position-specific SIA coverage.

    The INPUTS table remains the authority.  SIA is used only to avoid
    downloading exact inputs that do not touch any of the five validation
    positions, and its mapping is cached for deterministic offline reuse.
    """

    records = [dict(item) for item in snapshot["exposures"]]
    requested_detectors = {
        int(item["product"].detector)
        for item in products
        if item["product"].detector is not None
    }
    for record in records:
        detector = record.get("detector")
        if requested_detectors and detector is not None and int(detector) not in requested_detectors:
            record["skip_reason"] = "detector not requested by the mosaic image set"

    remote = [
        item for item in records
        if item.get("access_url") is not None and item.get("skip_reason") is None
    ]
    if not remote:
        snapshot = dict(snapshot)
        snapshot["exposures"] = records
        snapshot["coverage_mapping"] = "not-required"
        return snapshot

    cache_path = _coverage_cache_path(
        remote,
        sample_skies,
        snapshot.get("data_collection") or config.data_collection,
        config,
    )
    mapping = None
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        mapping = {
            str(url): tuple(int(value) for value in indices)
            for url, indices in cached.get("sample_indices_by_url", {}).items()
        }
    elif config.network_policy == "cache":
        queried = _query_irsa_exposures(
            sample_skies,
            snapshot.get("data_collection") or config.data_collection,
        )
        mapping = {
            _url_identity(item["access_url"]): tuple(
                sorted({int(value) for value in item.get("_query_sample_indices", ())})
            )
            for item in queried
            if item.get("access_url") is not None
        }
        _write_json_atomic(cache_path, {
            "schema_version": 1,
            "kind": "spherex_sample_coverage",
            "sample_indices_by_url": mapping,
        })

    snapshot = dict(snapshot)
    if mapping is None:
        snapshot["coverage_mapping"] = "unavailable"
        snapshot["exposures"] = records
        return snapshot

    matched = 0
    for record in records:
        if record.get("skip_reason") is not None:
            continue
        url = record.get("access_url")
        if url is None:
            continue
        indices = mapping.get(_url_identity(url), ())
        if indices:
            record["_sample_indices"] = list(indices)
            matched += 1
        else:
            record["skip_reason"] = "exact input does not cover a validation sample"
    if not matched and any(item.get("access_url") is not None for item in records):
        raise ValueError(
            "none of the exact mosaic INPUTS exposures match the five SIA sample positions"
        )
    snapshot["coverage_mapping"] = "sia2-exact-intersection"
    snapshot["coverage_mapping_digest"] = _stable_digest(mapping)
    snapshot["exposures"] = records
    return snapshot


def _cutout_url(url, center, size_deg=0.05):
    parsed = urlparse(str(url))
    if parsed.hostname != "irsa.ipac.caltech.edu" or "/ibe/data/" not in parsed.path:
        return str(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({
        "center": f"{center.ra.degree:.10f},{center.dec.degree:.10f}",
        "size": f"{max(float(size_deg), 0.01):.8f}",
    })
    return urlunparse(parsed._replace(query=urlencode(query)))


def _combined_cutout_geometry(sample_skies, sample_indices):
    indices = tuple(sorted({int(value) for value in sample_indices}))
    selected = sample_skies[list(indices)]
    if len(indices) == 1:
        return selected[0], 0.05
    ra = np.unwrap(np.asarray(selected.ra.radian, dtype=float))
    dec = np.asarray(selected.dec.radian, dtype=float)
    center = SkyCoord(
        ra=float(np.mean(ra)) * u.rad,
        dec=float(np.mean(dec)) * u.rad,
        frame="icrs",
    )
    radius = float(np.max(center.separation(selected).degree))
    # The extra 0.05 degree provides detector-pixel halo around the outermost
    # validation positions while still issuing only one cutout per exposure.
    return center, max(0.05, 2.0 * radius + 0.05)


def _retryable_download_error(exc):
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status = None if response is None else response.status_code
        return status in {408, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, HTTPError):
        return int(exc.code) in {408, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, requests.RequestException):
        return True
    return isinstance(
        exc,
        (URLError, TimeoutError, ConnectionError, IncompleteRead, OSError),
    )


def _iter_download_chunks(url, timeout):
    connect_timeout = min(30.0, float(timeout))
    with requests.get(
        str(url),
        stream=True,
        timeout=(connect_timeout, float(timeout)),
        headers={"User-Agent": "Spec7DT/0.14"},
    ) as response:
        response.raise_for_status()
        yield from response.iter_content(chunk_size=1024 * 1024)


def _download_to_cache(
    url,
    config,
    *,
    expected_sha256=None,
    cancel_event=None,
):
    parsed = urlparse(str(url))
    if parsed.scheme != "https" or parsed.hostname not in _IRSA_HOSTS:
        raise ValueError(f"remote SPHEREx URL is not an approved IRSA HTTPS host: {url}")
    download_root = config.resolved_cache_dir / "downloads"
    content_root = download_root / "sha256"
    index_root = download_root / "by-url"
    temporary_root = download_root / "temporary"
    for directory in (content_root, index_root, temporary_root):
        directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(parsed.path).suffix or ".fits"
    expected = None if expected_sha256 is None else str(expected_sha256).lower()
    if expected is not None:
        expected_path = content_root / f"{expected}{suffix}"
        if expected_path.is_file():
            digest = _file_sha256(expected_path)
            if digest.lower() != expected:
                raise ValueError(f"cached file checksum mismatch: {expected_path}")
            return expected_path, digest
    index_path = index_root / f"{sha256(str(url).encode('utf-8')).hexdigest()}.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index.get("url") != str(url):
            raise ValueError(f"cached URL index changed unexpectedly: {index_path}")
        digest = str(index.get("sha256", "")).lower()
        content_path = content_root / f"{digest}{index.get('suffix', suffix)}"
        if not content_path.is_file() or _file_sha256(content_path).lower() != digest:
            raise ValueError(f"cached file checksum mismatch: {content_path}")
        if expected is not None and digest != expected:
            raise ValueError(
                f"cached URL checksum {digest} does not match requested checksum {expected}"
            )
        return content_path, digest
    if config.network_policy != "cache":
        raise FileNotFoundError(
            f"remote exposure is not cached and network_policy='offline': {url}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="spherex-",
        suffix=".part",
        dir=temporary_root,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        attempts = int(config.download_retries) + 1
        for attempt in range(attempts):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("SPHEREx download cancelled after another failure")
            try:
                with temporary.open("wb") as stream:
                    for chunk in _iter_download_chunks(
                        url,
                        float(config.download_timeout_seconds),
                    ):
                        if cancel_event is not None and cancel_event.is_set():
                            raise CancelledError(
                                "SPHEREx download cancelled after another failure"
                            )
                        if not chunk:
                            continue
                        stream.write(chunk)
                break
            except CancelledError:
                raise
            except Exception as exc:
                if not _retryable_download_error(exc) or attempt + 1 >= attempts:
                    raise RuntimeError(
                        f"SPHEREx download failed after {attempt + 1} attempt(s): {url}: {exc}"
                    ) from exc
                delay = min(
                    30.0,
                    float(config.retry_backoff_seconds) * (2.0 ** attempt),
                )
                if cancel_event is not None:
                    if cancel_event.wait(delay):
                        raise CancelledError(
                            "SPHEREx download cancelled during retry backoff"
                        )
                elif delay:
                    Event().wait(delay)
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("SPHEREx download cancelled after another failure")
        digest = _file_sha256(temporary)
        if expected is not None and digest.lower() != expected:
            raise ValueError(f"downloaded file checksum mismatch: {url}")
        content_path = content_root / f"{digest}{suffix}"
        if content_path.is_file():
            if _file_sha256(content_path).lower() != digest:
                raise ValueError(f"cached file checksum mismatch: {content_path}")
            temporary.unlink()
        else:
            temporary.replace(content_path)
        _write_json_atomic(index_path, {
            "url": str(url),
            "sha256": digest,
            "suffix": suffix,
        })
    finally:
        if temporary.exists():
            temporary.unlink()
    return content_path, digest


def _materialize_one(record, config, sample_skies, cancel_event=None):
    item = dict(record)
    if item.get("skip_reason") is not None:
        return item
    if item.get("path") is not None:
        path = Path(item["path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Level-2 exposure is missing: {path}")
        digest = _file_sha256(path)
        if item.get("sha256") and digest.lower() != str(item["sha256"]).lower():
            raise ValueError(f"Level-2 exposure checksum mismatch: {path}")
    else:
        cutouts = []
        original_url = str(item["access_url"])
        exposure_index = (
            config.resolved_cache_dir
            / "downloads"
            / "by-exposure"
            / f"{sha256(str(item['exposure_id']).encode('utf-8')).hexdigest()}.json"
        )
        if exposure_index.is_file():
            pinned = json.loads(exposure_index.read_text(encoding="utf-8"))
            if pinned.get("access_url") != original_url:
                raise ValueError(
                    f"access_url changed for exposure {item['exposure_id']}: "
                    f"{pinned.get('access_url')} != {original_url}"
                )
        sample_indices = item.pop("_sample_indices", None)
        if sample_indices is None:
            sample_indices = item.pop("_query_sample_indices", None)
        assigned = None if sample_indices is None else tuple(
            sorted({int(value) for value in sample_indices})
        )
        if assigned is None:
            assigned = tuple(range(len(sample_skies)))
        center, size_deg = _combined_cutout_geometry(sample_skies, assigned)
        plans = [(assigned, center, size_deg)]
        for indices, center, size_deg in plans:
            url = _cutout_url(original_url, center, size_deg)
            path, digest = _download_to_cache(
                url,
                config,
                expected_sha256=(item.get("sha256") if url == original_url else None),
                cancel_event=cancel_event,
            )
            cutout = {
                "sample_indices": list(indices),
                "path": str(path),
                "sha256": digest,
                "resolved_url": url,
            }
            if len(indices) == 1:
                cutout["sample_index"] = int(indices[0])
            cutouts.append(cutout)
        item["source_sha256"] = item.pop("sha256", None)
        item["cutouts"] = cutouts
        _write_json_atomic(exposure_index, {
            "exposure_id": item["exposure_id"],
            "access_url": original_url,
            "cutouts": [
                {
                    "sample_indices": value["sample_indices"],
                    "sha256": value["sha256"],
                    "resolved_url": value["resolved_url"],
                }
                for value in cutouts
            ],
        })
        return item
    item["path"] = str(path)
    item["sha256"] = digest
    return item


def _materialize_exposures(records, config, sample_skies):
    skipped = [dict(item) for item in records if item.get("skip_reason") is not None]
    active = [dict(item) for item in records if item.get("skip_reason") is None]
    if not active:
        return sorted(skipped, key=lambda item: item["exposure_id"])

    progress = None
    task_id = None
    if config.progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=2,
            transient=False,
        )
        progress.start()
        task_id = progress.add_task(
            "SPHEREx PSF Level-2 downloads",
            total=len(active),
        )

    cancel_event = Event()
    executor = ThreadPoolExecutor(max_workers=config.max_workers)
    iterator = iter(active)
    pending = {}
    results = list(skipped)
    failed = False

    def submit_next():
        try:
            record = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(
            _materialize_one,
            record,
            config,
            sample_skies,
            cancel_event,
        )
        pending[future] = record
        return True

    try:
        for _ in range(min(config.max_workers, len(active))):
            submit_next()
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                record = pending.pop(future)
                try:
                    result = future.result()
                except Exception:
                    failed = True
                    cancel_event.set()
                    for remaining in pending:
                        remaining.cancel()
                    raise
                results.append(result)
                if progress is not None:
                    progress.update(
                        task_id,
                        description=(
                            "SPHEREx PSF Level-2 downloads: "
                            f"{record['exposure_id']}"
                        ),
                        advance=1,
                    )
                submit_next()
    except BaseException:
        failed = True
        cancel_event.set()
        for remaining in pending:
            remaining.cancel()
        raise
    finally:
        if failed:
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True, cancel_futures=True)
        if progress is not None:
            progress.stop()
    return sorted(results, key=lambda item: item["exposure_id"])


def _expand_materialized_exposures(records):
    expanded = []
    for record in records:
        if record.get("skip_reason") is not None:
            continue
        cutouts = record.get("cutouts") or ()
        if not cutouts:
            expanded.append(dict(record))
            continue
        for cutout in cutouts:
            item = dict(record)
            item.pop("cutouts", None)
            item.update(cutout)
            indices = cutout.get("sample_indices")
            if indices is None and cutout.get("sample_index") is not None:
                indices = [int(cutout["sample_index"])]
            item["_sample_indices"] = [int(value) for value in indices or ()]
            expanded.append(item)
    return expanded


def _materialize_calibration_registry(value, config):
    if value is None:
        return None
    base = None
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        path = Path(value).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent
    entries = []
    for entry in raw.get("calibrations", ()):
        item = dict(entry)
        path_value = item.get("path")
        url = item.get("access_url") or item.get("url")
        if path_value is not None and str(path_value).startswith("https://"):
            url = str(path_value)
            path_value = None
        if path_value is not None:
            path = Path(path_value).expanduser()
            if not path.is_absolute() and base is not None:
                path = base / path
            item["path"] = str(path.resolve())
        elif url is not None:
            raise ValueError(
                "automatic ePSF calibration download is disabled; provide a local "
                "checksum-pinned calibration registry path"
            )
        else:
            raise ValueError("calibration registry entry requires path or access_url")
        entries.append(item)
    return {"schema_version": int(raw.get("schema_version", 1)), "calibrations": entries}


def _wavelength_interpolator(hdul):
    if "WCS-WAVE" not in hdul:
        raise ValueError("Level-2 exposure has no WCS-WAVE extension")
    table = hdul["WCS-WAVE"].data
    if table is None or len(table) != 1:
        raise ValueError("Level-2 WCS-WAVE must contain exactly one row")
    names = {str(name).upper(): str(name) for name in (table.dtype.names or ())}
    if not {"X", "Y", "VALUES"}.issubset(names):
        raise ValueError("Level-2 WCS-WAVE is missing X, Y, or VALUES")
    x = np.asarray(table[names["X"]][0], dtype=float).ravel()
    y = np.asarray(table[names["Y"]][0], dtype=float).ravel()
    values = np.asarray(table[names["VALUES"]][0], dtype=float)
    if values.size % (x.size * y.size) != 0:
        raise ValueError("Level-2 WCS-WAVE VALUES dimensions are inconsistent")
    components = values.size // (x.size * y.size)
    if values.shape == (y.size, x.size, components):
        pass
    elif values.shape == (components, y.size, x.size):
        values = np.moveaxis(values, 0, -1)
    else:
        values = values.reshape(y.size, x.size, components)
    if components < 2 or not np.all(np.isfinite(values[..., :2])):
        raise ValueError("Level-2 WCS-WAVE has invalid wavelength samples")
    unit = hdul["WCS-WAVE"].columns[names["VALUES"]].unit
    if unit:
        try:
            values = values * u.Unit(unit).to(u.um)
        except Exception as exc:
            raise ValueError(f"Level-2 WCS-WAVE VALUES unit is not wavelength: {unit}") from exc
    interpolator = RegularGridInterpolator(
        (y, x),
        values[..., 0],
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    # WCS-WAVE control points use the FITS 1-based pixel convention, while
    # astropy ``world_to_pixel`` returns zero-based array coordinates.
    return lambda px, py: float(interpolator((float(py) + 1.0, float(px) + 1.0)))


def _merged_header(primary, image):
    result = primary.copy()
    result.update(image)
    return result


def _read_exposure(record, calibration_registry, epsf_cache=None):
    path = Path(record["path"]).resolve()
    with fits.open(path, memmap=False) as hdul:
        if "IMAGE" not in hdul:
            raise ValueError(f"Level-2 exposure has no IMAGE extension: {path}")
        image_hdu = hdul["IMAGE"]
        image = np.asarray(image_hdu.data, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"Level-2 IMAGE is not 2-D: {path}")
        flags = (
            np.asarray(hdul["FLAGS"].data, dtype=np.int64)
            if "FLAGS" in hdul and hdul["FLAGS"].data is not None
            else np.zeros(image.shape, dtype=np.int64)
        )
        if flags.shape != image.shape:
            raise ValueError(f"Level-2 FLAGS shape does not match IMAGE: {path}")
        primary = hdul[0].header.copy()
        image_header = _merged_header(primary, image_hdu.header)
        wavelength = _wavelength_interpolator(hdul)
        detector = image_header.get("DETECTOR")
        if detector is None:
            raise ValueError(f"Level-2 exposure has no DETECTOR keyword: {path}")
        version = image_header.get("VERSION")
        major = _version_major(version)
        embedded = None
        if "EPSF" in hdul:
            embedded = {"type": "oversampled_table", "hdu": "EPSF", "usable": True}
        elif "PSF" in hdul:
            embedded = {"type": "legacy_psf_cube", "hdu": "PSF", "usable": False}
        product = ImageProductMetadata(
            product_kind=(
                "cutout" if "CRPIX1A" in image_header and "CRPIX2A" in image_header
                else "exposure"
            ),
            facility="SPHEREx",
            release="qr3" if major is not None and major >= 7 else "qr2",
            version=None if version is None else str(version),
            detector=detector,
            source_path=str(path),
            hdu="IMAGE",
            sampling=sampling_from_header(image_header),
            embedded_psf=embedded,
        )
        if product.sampling is None:
            raise ValueError(f"Level-2 exposure has no angular WCS sampling: {path}")
        if record.get("detector") is not None and int(record["detector"]) != int(detector):
            raise ValueError(
                f"exposure detector mismatch for {record['exposure_id']}: "
                f"list={record['detector']} file={detector}"
            )
        expected_release = str(record.get("release") or "").lower()
        if expected_release and product.release not in expected_release and expected_release not in product.release:
            raise ValueError(
                f"exposure release mismatch for {record['exposure_id']}: "
                f"list={expected_release} file={product.release}"
            )
        resolver_config = type("CalibrationConfig", (), {
            "calibration_registry": calibration_registry,
        })()
        epsf_path, epsf_hdu, _, _ = SpherexEPSFProvider._calibration_path(
            product,
            resolver_config,
        )
        if Path(epsf_path).resolve() == path and "EPSF" in hdul:
            epsf_identity = (
                "embedded",
                str(product.detector),
                str(product.release),
                str(product.version),
                repr(hdul["EPSF"].header.get("HISTORY")),
                int(hdul["EPSF"].header.get("NAXIS2", 0)),
                int(hdul["EPSF"].header.get("OVSMPX", 1)),
                int(hdul["EPSF"].header.get("OVSMPY", 1)),
            )
        else:
            epsf_identity = (
                "calibration",
                str(Path(epsf_path).resolve()),
                str(epsf_hdu),
            )
        cached_epsf = None if epsf_cache is None else epsf_cache.get(epsf_identity)
        if cached_epsf is None:
            cached_epsf = SpherexEPSFProvider._load_table(epsf_path, epsf_hdu)
            if epsf_cache is not None:
                epsf_cache[epsf_identity] = cached_epsf
        epsf_table, epsf_header, epsf_names = cached_epsf
        calibration_detector = epsf_header.get("DETECTOR")
        if (
            calibration_detector is not None
            and str(calibration_detector).strip() != str(detector).strip()
        ):
            raise ValueError(
                f"ePSF detector mismatch for {record['exposure_id']}: "
                f"image={detector} calibration={calibration_detector}"
            )
    return _ExposureData(
        record=dict(record),
        path=path,
        primary_header=primary,
        image_header=image_header,
        image=image,
        flags=flags,
        wcs=WCS(image_header, relax=True).celestial,
        sampling=product.sampling,
        wavelength=wavelength,
        product=product,
        epsf_table=epsf_table,
        epsf_header=epsf_header,
        epsf_names=epsf_names,
    )


def _parent_position(exposure, x, y):
    if exposure.product.product_kind == "cutout":
        return (
            1.0 + x - float(exposure.image_header["CRPIX1A"]),
            1.0 + y - float(exposure.image_header["CRPIX2A"]),
        )
    return x, y


def _render_exposure_epsf(exposure, x, y):
    names = exposure.epsf_names
    table = exposure.epsf_table
    parent_x, parent_y = _parent_position(exposure, x, y)
    distances = np.hypot(
        parent_x - np.asarray(table[names["XCENTER"]], dtype=float),
        parent_y - np.asarray(table[names["YCENTER"]], dtype=float),
    )
    row_index = int(np.nanargmin(distances))
    data = np.asarray(table[names["EPSF"]][row_index], dtype=float)
    if data.ndim != 2 or not np.all(np.isfinite(data)):
        raise ValueError(f"invalid ePSF row {row_index} in {exposure.path}")
    peak = float(np.max(data))
    negative_mass = float(-np.sum(np.minimum(data, 0.0), dtype=np.float64))
    positive_mass = float(np.sum(np.maximum(data, 0.0), dtype=np.float64))
    negative_fraction = negative_mass / positive_mass if positive_mass > 0 else np.inf
    if peak <= 0 or float(np.min(data)) < -_NEGATIVE_PEAK_TOLERANCE * peak:
        raise ValueError(f"ePSF row {row_index} has significant negative samples")
    if negative_fraction > _NEGATIVE_MASS_TOLERANCE:
        raise ValueError(f"ePSF row {row_index} has excessive negative mass")
    data = np.maximum(data, 0.0)
    oy = int(exposure.epsf_header.get("OVSMPY", 1))
    ox = int(exposure.epsf_header.get("OVSMPX", 1))
    center_x = int(np.floor(float(x) + 0.5))
    center_y = int(np.floor(float(y) + 0.5))
    phase = (float(y) - center_y, float(x) - center_x)
    rendered = render_oversampled_psf(
        data,
        (oy, ox),
        phase,
        exposure.sampling,
        stamp_size=_LEVEL2_STAMP_SIZE,
        integration_subsamples=min(8, max(oy, ox)),
    )
    return rendered, center_x, center_y, row_index, phase, negative_fraction


def _tile_bounds(group_id, metadata, image_shape, config):
    if config.tile_bounds:
        if str(group_id) in config.tile_bounds:
            bounds = config.tile_bounds[str(group_id)]
        elif str(metadata.product_id) in config.tile_bounds:
            bounds = config.tile_bounds[str(metadata.product_id)]
        else:
            bounds = None
        if bounds is not None:
            return tuple(map(float, bounds))
    ny, nx = map(int, image_shape)
    return 0.0, 0.0, float(nx - 1), float(ny - 1)


def _sample_positions(bounds):
    x0, y0, x1, y1 = bounds
    rounded_x0 = float(np.floor(x0 + 0.5))
    rounded_y0 = float(np.floor(y0 + 0.5))
    rounded_x1 = float(np.floor(x1 + 0.5))
    rounded_y1 = float(np.floor(y1 + 0.5))
    return [
        (float(np.floor((x0 + x1) / 2.0 + 0.5)), float(np.floor((y0 + y1) / 2.0 + 0.5))),
        (rounded_x0, rounded_y0),
        (rounded_x1, rounded_y0),
        (rounded_x0, rounded_y1),
        (rounded_x1, rounded_y1),
    ]


def _output_stamp_size(exposures, output_sampling):
    output_scale = output_sampling.equivalent_scale_arcsec
    native_scale = max(item.sampling.equivalent_scale_arcsec for item in exposures)
    size = int(np.ceil(_LEVEL2_STAMP_SIZE * native_scale / output_scale * np.sqrt(2.0))) + 6
    size = max(15, size)
    return size + 1 - size % 2


def _synthesize_sample(
    exposures,
    output_header,
    output_sampling,
    position,
    wl_min,
    wl_max,
    nhits_plane,
    source,
    sample_index=None,
):
    output_wcs = WCS(output_header, relax=True).celestial
    sky = output_wcs.pixel_to_world(float(position[0]), float(position[1]))
    size = _output_stamp_size(exposures, output_sampling)
    half = size // 2
    flux = np.zeros((size, size), dtype=float)
    hits = np.zeros((size, size), dtype=np.int64)
    contributors = []
    edge_distances = []
    rows = []
    phases = []
    negative_fractions = []
    for exposure in exposures:
        assigned_samples = exposure.record.get("_sample_indices")
        assigned_sample = exposure.record.get("_sample_index")
        if (
            sample_index is not None
            and assigned_samples is not None
            and int(sample_index) not in {int(value) for value in assigned_samples}
        ):
            continue
        if (
            assigned_samples is None
            and assigned_sample is not None
            and sample_index is not None
            and int(assigned_sample) != int(sample_index)
        ):
            continue
        x, y = exposure.wcs.world_to_pixel(sky)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if (
            x < -0.5 or y < -0.5
            or x > exposure.image.shape[1] - 0.5
            or y > exposure.image.shape[0] - 0.5
        ):
            continue
        wavelength = exposure.wavelength(x, y)
        if not np.isfinite(wavelength) or wavelength < wl_min or wavelength > wl_max:
            continue
        edge_distances.append(min(wavelength - wl_min, wl_max - wavelength))
        rendered, center_x, center_y, row_index, phase, negative_fraction = (
            _render_exposure_epsf(exposure, x, y)
        )
        yy, xx = np.indices(rendered.shape)
        native_x = center_x + xx - rendered.shape[1] // 2
        native_y = center_y + yy - rendered.shape[0] // 2
        valid = (
            (native_x >= 0) & (native_x < exposure.image.shape[1])
            & (native_y >= 0) & (native_y < exposure.image.shape[0])
        )
        if not np.any(valid):
            continue
        vx = native_x[valid].astype(int)
        vy = native_y[valid].astype(int)
        valid_values = np.isfinite(exposure.image[vy, vx])
        valid_values &= (exposure.flags[vy, vx] & _DEFAULT_REJECT_FLAG_MASK) == 0
        if not np.any(valid_values):
            continue
        vx = vx[valid_values]
        vy = vy[valid_values]
        values = rendered[valid][valid_values]
        world = exposure.wcs.pixel_to_world(vx.astype(float), vy.astype(float))
        ox, oy = output_wcs.world_to_pixel(world)
        local_x = np.floor(ox - position[0] + 0.5).astype(int) + half
        local_y = np.floor(oy - position[1] + 0.5).astype(int) + half
        inside = (
            np.isfinite(ox) & np.isfinite(oy)
            & (local_x >= 0) & (local_x < size)
            & (local_y >= 0) & (local_y < size)
        )
        if not np.any(inside):
            continue
        np.add.at(flux, (local_y[inside], local_x[inside]), values[inside])
        np.add.at(hits, (local_y[inside], local_x[inside]), 1)
        contributors.append(exposure.record["exposure_id"])
        rows.append(row_index)
        phases.append(phase)
        negative_fractions.append(negative_fraction)
    if not contributors:
        raise ValueError("no Level-2 exposure contributes at this band/tile position")
    coadd = np.divide(flux, hits, out=np.zeros_like(flux), where=hits > 0)
    total = float(np.sum(coadd, dtype=np.float64))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("equal-hit PSF coadd produced no positive flux")
    coadd /= total
    center_hits = int(hits[half, half])
    hit_validation = "unavailable"
    expected_hits = None
    if nhits_plane is not None:
        ix = int(np.floor(position[0] + 0.5))
        iy = int(np.floor(position[1] + 0.5))
        raw_expected = float(nhits_plane[iy, ix])
        if not np.isfinite(raw_expected) or abs(raw_expected - round(raw_expected)) > 1e-6:
            raise ValueError(f"mosaic NHITS is not a finite integer at {position}")
        expected_hits = int(round(raw_expected))
        hit_validation = "matched" if center_hits == expected_hits else "mismatch"
        if center_hits != expected_hits:
            raise ValueError(
                f"NHITS mismatch at {position}: synthesized={center_hits}, mosaic={expected_hits}"
            )
    model = validate_effective_psf(
        coadd,
        output_sampling,
        source,
        edge_tolerance=_PSF_EDGE_TOLERANCE,
        negative_peak_tolerance=_NEGATIVE_PEAK_TOLERANCE,
        negative_mass_tolerance=_NEGATIVE_MASS_TOLERANCE,
        centroid_limit=1.0,
        frame="image",
        provenance={
            "exposure_count": len(set(contributors)),
            "exposure_ids": sorted(set(contributors)),
            "epsf_rows": rows,
            "phases": phases,
            "hit_validation": hit_validation,
            "synthesized_center_hits": center_hits,
            "mosaic_center_hits": expected_hits,
            "minimum_band_edge_distance_um": float(min(edge_distances)),
            "maximum_negative_mass_fraction": float(max(negative_fractions)),
        },
    )
    return model, {
        "exposure_count": len(set(contributors)),
        "hit_validation": hit_validation,
        "synthesized_center_hits": center_hits,
        "mosaic_center_hits": expected_hits,
        "minimum_band_edge_distance_um": float(min(edge_distances)),
    }


def _load_nhits(metadata):
    hdu_name = metadata.provenance.get("nhits_hdu")
    if not hdu_name or metadata.source_path is None:
        return None
    path = Path(metadata.source_path).expanduser()
    if not path.is_file():
        return None
    with fits.open(path, memmap=False) as hdul:
        if hdu_name not in hdul or hdul[hdu_name].data is None:
            return None
        data = np.asarray(hdul[hdu_name].data)
    if data.ndim == 3:
        data = data[int(metadata.plane)]
    if data.ndim != 2:
        raise ValueError("mosaic NHITS must be a 2-D plane or 3-D cube")
    return np.asarray(data)


def _write_psf_artifact(path, representative, samples):
    matrix = np.asarray(representative.sampling.matrix_arcsec, dtype=float)
    header = fits.Header()
    header["PSCALE"] = representative.pixel_scale_arcsec
    header["PSFM11"] = matrix[0, 0]
    header["PSFM12"] = matrix[0, 1]
    header["PSFM21"] = matrix[1, 0]
    header["PSFM22"] = matrix[1, 1]
    header["PIXINT"] = True
    hdus = [fits.PrimaryHDU(np.asarray(representative.data, dtype=np.float32), header=header)]
    for index, sample in enumerate(samples, start=1):
        sample_header = header.copy()
        sample_header["POSX"] = float(sample.position[0])
        sample_header["POSY"] = float(sample.position[1])
        hdus.append(fits.ImageHDU(
            np.asarray(sample.model.data, dtype=np.float32),
            header=sample_header,
            name=f"SAMPLE{index}",
        ))
    fits.HDUList(hdus).writeto(path, overwrite=False, checksum=True)


def _group_products(image_set):
    groups = {}
    for galaxy, observatories in image_set.data.items():
        for observatory, bands in observatories.items():
            for band, image in bands.items():
                product = (
                    image_set.product_metadata
                    .get(galaxy, {})
                    .get(observatory, {})
                    .get(band)
                )
                if product is None:
                    continue
                product = ImageProductMetadata.from_value(product)
                if (
                    str(product.facility or observatory).lower() != "spherex"
                    or product.product_kind != "coadd"
                    or not product.requires_output_effective_psf
                    # PSF-homogenized cubes already carry their output PSF;
                    # synthesizing from INPUTS would give the pre-matching PSF.
                    or dict(product.provenance or {}).get("psf_homogenized")
                ):
                    continue
                group_id = product.product_group_id or _stable_digest({
                    "source_path": product.source_path,
                    "hdu": product.hdu,
                })
                groups.setdefault(str(group_id), []).append({
                    "galaxy": galaxy,
                    "observatory": observatory,
                    "band": band,
                    "image": image,
                    "header": image_set.header[galaxy][observatory][band],
                    "product": product,
                })
    return groups


def _absolute_manifest(value):
    if value is None:
        return {"schema_version": 2, "models": []}
    base = None
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        path = Path(value).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent
    entries = raw.get("models", raw)
    if isinstance(entries, Mapping):
        normalized_entries = []
        for key, entry_value in entries.items():
            entry = (
                {"path": str(entry_value)}
                if isinstance(entry_value, (str, Path))
                else dict(entry_value)
            )
            if "." in str(key):
                facility, channel = str(key).split(".", 1)
                entry.setdefault("observatory", facility)
                entry.setdefault("band", channel)
            normalized_entries.append(entry)
        entries = normalized_entries
    result = []
    for entry_value in entries:
        entry = dict(entry_value)
        representation = dict(entry.get("representation", {}))
        for container in (entry, representation):
            if container.get("path") is not None:
                path = Path(container["path"]).expanduser()
                if not path.is_absolute() and base is not None:
                    path = base / path
                container["path"] = str(path.resolve())
        entry["representation"] = representation
        validation = dict(entry.get("spatial_validation", {}))
        samples = []
        for sample_value in validation.get("samples", ()):
            sample = dict(sample_value)
            if sample.get("path") is not None:
                path = Path(sample["path"]).expanduser()
                if not path.is_absolute() and base is not None:
                    path = base / path
                sample["path"] = str(path.resolve())
            samples.append(sample)
        if samples:
            validation["samples"] = samples
            entry["spatial_validation"] = validation
        result.append(entry)
    return {"schema_version": 2, "models": result}


def _entry_matches_product(entry, product, observatory, band, tile_bounds=None):
    selector = dict(entry.get("selector", {}))
    selector.setdefault("facility", entry.get("observatory"))
    selector.setdefault("channel", entry.get("band"))
    checks = (
        ("product_id", product.product_id),
        ("facility", product.facility or observatory),
        ("channel", product.channel or band),
    )
    for key, actual in checks:
        expected = selector.get(key)
        if expected is not None and str(expected).lower() != str(actual).lower():
            return False
    region = selector.get("valid_region")
    if region is not None and tile_bounds is not None:
        x0, y0, x1, y1 = map(float, region)
        tx0, ty0, tx1, ty1 = map(float, tile_bounds)
        if tx0 < x0 or ty0 < y0 or tx1 > x1 or ty1 > y1:
            return False
    return any(selector.get(key) is not None for key in ("product_id", "facility", "channel"))


def _manifest_covers_product(manifest, product, observatory, band, tile_bounds=None):
    entries = _absolute_manifest(manifest)["models"]
    matches = [
        entry
        for entry in entries
        if _entry_matches_product(entry, product, observatory, band, tile_bounds)
    ]
    if not matches:
        return False

    def specificity(entry):
        selector = dict(entry.get("selector", {}))
        selector.setdefault("facility", entry.get("observatory"))
        selector.setdefault("channel", entry.get("band"))
        return (
            (100 if selector.get("product_id") is not None else 0)
            + (10 if selector.get("valid_region") is not None else 0)
            + sum(selector.get(key) is not None for key in ("facility", "channel"))
        )

    maximum = max(specificity(entry) for entry in matches)
    best_count = sum(specificity(entry) == maximum for entry in matches)
    if best_count > 1:
        raise ValueError(
            f"existing PSF manifest is ambiguous for product {product.product_id} "
            f"({observatory}/{band})"
        )
    return True


def spherex_mosaic_synthesis_required(image_set, manifest=None, tile_bounds=None):
    """Return whether at least one SPHEREx coadd lacks a manifest PSF."""

    for group_id, products in _group_products(image_set).items():
        for item in products:
            product = item["product"]
            bounds = None
            if tile_bounds:
                bounds = tile_bounds.get(str(group_id), tile_bounds.get(str(product.product_id)))
            if not _manifest_covers_product(
                manifest,
                product,
                item["observatory"],
                item["band"],
                bounds,
            ):
                return True
    return False


def _axis_size_pixels(value, scale_arcsec):
    if isinstance(value, u.Quantity):
        if not value.unit.is_equivalent(u.arcsec):
            raise TypeError("angular synthesis footprints must be convertible to arcsec")
        return float(value.to_value(u.arcsec)) / float(scale_arcsec)
    return float(value)


def _shape_from_size(value, sampling):
    if value is None:
        return None
    if isinstance(value, u.Quantity) and value.isscalar:
        values = (value, value)
    elif np.isscalar(value):
        values = (value, value)
    else:
        values = tuple(value)
        if len(values) != 2:
            raise ValueError("trim footprint must be scalar or contain (ny, nx)")
    matrix = np.asarray(sampling.matrix_arcsec, dtype=float)
    x_scale, y_scale = np.sqrt(np.sum(matrix ** 2, axis=0))
    ny = _axis_size_pixels(values[0], y_scale)
    nx = _axis_size_pixels(values[1], x_scale)
    if not np.all(np.isfinite((ny, nx))) or ny <= 0 or nx <= 0:
        raise ValueError("trim footprint must have finite positive dimensions")
    return ny, nx


def synthesis_config_for_analysis_footprint(
    image_set,
    config,
    *,
    pretrim_fov=None,
    trim_size=None,
    metadata_resolver=None,
):
    """Fill unspecified tile bounds from the non-mutating future analysis FoV."""

    config = SpherexMosaicPSFSynthesisConfig.from_value(config)
    if config.tile_bounds is not None:
        return config
    groups = _group_products(image_set)
    bounds_by_group = {}
    for group_id, products in groups.items():
        item = products[0]
        product = item["product"]
        sampling = product.sampling or sampling_from_header(item["header"])
        if sampling is None:
            continue
        sizes = [
            size
            for size in (
                _shape_from_size(pretrim_fov, sampling),
                _shape_from_size(trim_size, sampling),
            )
            if size is not None
        ]
        if not sizes:
            continue
        ny = min(size[0] for size in sizes)
        nx = min(size[1] for size in sizes)
        shape = np.shape(item["image"])
        try:
            from ..utils.utility import useful_functions

            center = useful_functions.get_sky_loc(
                item["galaxy"],
                header=item["header"],
                metadata_resolver=metadata_resolver,
                required=True,
                image_shape=shape,
            )
            cx, cy = WCS(item["header"], relax=True).celestial.world_to_pixel(center)
        except Exception as exc:
            raise ValueError(
                f"cannot determine synthesis footprint center for {item['galaxy']}: {exc}"
            ) from exc
        x0 = max(0.0, float(cx) - 0.5 * (nx - 1.0))
        y0 = max(0.0, float(cy) - 0.5 * (ny - 1.0))
        x1 = min(float(shape[1] - 1), float(cx) + 0.5 * (nx - 1.0))
        y1 = min(float(shape[0] - 1), float(cy) + 0.5 * (ny - 1.0))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                f"analysis footprint does not overlap SPHEREx mosaic group {group_id}"
            )
        bounds_by_group[str(group_id)] = (x0, y0, x1, y1)
    return replace(config, tile_bounds=bounds_by_group or None)


def _selector_matches_generated(existing, generated):
    selector = dict(existing.get("selector", {}))
    selector.setdefault("facility", existing.get("observatory"))
    selector.setdefault("channel", existing.get("band"))
    target = dict(generated.get("selector", {}))
    for key in ("product_id", "facility", "channel"):
        if selector.get(key) is not None and str(selector[key]).lower() != str(target.get(key)).lower():
            return False
    return any(selector.get(key) is not None for key in ("product_id", "facility", "channel"))


def merge_psf_manifests(existing, generated):
    """Merge manifests while preserving every matching user-provided model."""

    first = _absolute_manifest(existing)
    second = _absolute_manifest(generated)
    models = list(first["models"])
    for entry in second["models"]:
        if any(_selector_matches_generated(item, entry) for item in models):
            continue
        models.append(entry)
    return {"schema_version": 2, "models": models}


def build_spherex_mosaic_psf_manifest(
    image_set,
    config: SpherexMosaicPSFSynthesisConfig | Mapping,
    *,
    calibration_registry=None,
    existing_manifest=None,
) -> SpherexMosaicPSFBuildResult:
    """Synthesize five-position mosaic PSFs and atomically publish a v2 manifest."""

    config = SpherexMosaicPSFSynthesisConfig.from_value(config)
    groups = _group_products(image_set)
    if existing_manifest is not None:
        uncovered = {}
        for group_id, products in groups.items():
            for item in products:
                bounds = _tile_bounds(
                    group_id,
                    item["product"],
                    np.shape(item["image"]),
                    config,
                )
                if not _manifest_covers_product(
                    existing_manifest,
                    item["product"],
                    item["observatory"],
                    item["band"],
                    bounds,
                ):
                    uncovered.setdefault(group_id, []).append(item)
        groups = uncovered
    if not groups:
        raise SpherexMosaicPSFBuildError({
            "input": "no uncovered SPHEREx coadd product requires PSF synthesis"
        })
    artifact_root = config.resolved_artifact_dir
    config.resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    calibration_registry = _materialize_calibration_registry(calibration_registry, config)

    group_inputs = {}
    global_failures = {}
    for group_id, products in groups.items():
        reference = products[0]
        bounds = _tile_bounds(
            group_id,
            reference["product"],
            np.shape(reference["image"]),
            config,
        )
        center, radius_deg, sample_skies = _footprint(
            reference["header"],
            np.shape(reference["image"]),
            bounds,
        )
        try:
            snapshot = _load_explicit_exposure_list(config.exposure_list, group_id)
            if snapshot is None:
                snapshot = _load_mosaic_inputs(
                    reference["product"],
                    group_id,
                    config,
                )
                if snapshot is not None:
                    snapshot = _assign_embedded_input_samples(
                        snapshot,
                        products,
                        sample_skies,
                        config,
                    )
                else:
                    if not config.query_if_missing:
                        raise ValueError(
                            "no explicit exposure list, no mosaic INPUTS HDU, and "
                            "query_if_missing=False"
                        )
                    records = _query_irsa_exposures(
                        sample_skies,
                        config.data_collection,
                    )
                    snapshot = {
                        "schema_version": 1,
                        "kind": "spherex_level2_exposure_list",
                        "product_group_id": group_id,
                        "selection": "candidate",
                        "data_collection": config.data_collection,
                        "source": "sia2-query",
                        "exposures": records,
                    }
            materialized = _materialize_exposures(
                snapshot["exposures"],
                config,
                sample_skies,
            )
            snapshot["exposures"] = materialized
            if snapshot.get("data_collection") is None:
                snapshot["data_collection"] = config.data_collection
            group_inputs[group_id] = {
                "products": products,
                "bounds": bounds,
                "snapshot": snapshot,
                "center": center,
                "radius_deg": radius_deg,
                "sample_skies": sample_skies,
            }
        except Exception as exc:
            global_failures[f"group/{group_id}"] = str(exc)
    if global_failures:
        raise SpherexMosaicPSFBuildError(global_failures)

    canonical_snapshot = {
        "schema_version": 1,
        "kind": "spherex_level2_exposure_lists",
        "groups": [group_inputs[key]["snapshot"] for key in sorted(group_inputs)],
    }
    recipe = {
        "method": "spherex_level2_epsf_equal_hit_v1",
        "sample_pattern": "center_corners",
        "wavelength_method": "embedded_wcs_wave",
        "coadd_method": "equal_hit_nearest_bin",
        "flag_policy": "irsa_mosaic_overflow_outlier",
        "data_collection": config.data_collection,
        "tile_bounds": config.tile_bounds,
        "calibration_registry_digest": (
            None if calibration_registry is None else _stable_digest(calibration_registry)
        ),
    }
    build_digest = _stable_digest({
        "snapshot": canonical_snapshot,
        "recipe": recipe,
        "products": sorted(
            item["product"].product_id
            for group in group_inputs.values()
            for item in group["products"]
        ),
    })
    final_dir = artifact_root / build_digest
    manifest_path = final_dir / "psf_manifest.json"
    snapshot_path = final_dir / "exposures.json"
    qc_path = final_dir / "qc.json"
    if manifest_path.is_file() and snapshot_path.is_file() and qc_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        artifacts = tuple(sorted(final_dir.glob("*.fits")))
        return SpherexMosaicPSFBuildResult(
            manifest,
            manifest_path,
            snapshot_path,
            artifacts,
            qc,
        )
    if final_dir.exists():
        raise SpherexMosaicPSFBuildError({
            "cache": f"generated artifact directory is incomplete: {final_dir}"
        })

    temporary = Path(tempfile.mkdtemp(prefix=f".{build_digest}.", dir=artifact_root))
    failures = {}
    manifest_models = []
    artifact_names = []
    qc = {"build_digest": build_digest, "recipe": recipe, "groups": {}}
    epsf_cache = {}
    try:
        _write_json_atomic(temporary / "exposures.json", canonical_snapshot)
        for group_id, group in group_inputs.items():
            records = _expand_materialized_exposures(group["snapshot"]["exposures"])
            try:
                exposures = [
                    _read_exposure(item, calibration_registry, epsf_cache)
                    for item in records
                ]
            except Exception as exc:
                failures[f"group/{group_id}/exposures"] = str(exc)
                continue
            group_qc = {
                "selection": group["snapshot"]["selection"],
                "exposure_source": group["snapshot"].get("source", "explicit"),
                "listed_exposure_count": len(group["snapshot"]["exposures"]),
                "skipped_exposure_count": sum(
                    item.get("skip_reason") is not None
                    for item in group["snapshot"]["exposures"]
                ),
                "exposure_count": len({
                    item.record["exposure_id"] for item in exposures
                }),
                "level2_file_count": len(exposures),
                "bands": {},
            }
            qc["groups"][group_id] = group_qc
            for item in group["products"]:
                product = item["product"]
                band = item["band"]
                key = f"{item['galaxy']}/{item['observatory']}/{band}"
                try:
                    output_sampling = product.sampling or sampling_from_header(item["header"])
                    if output_sampling is None:
                        raise ValueError("mosaic output has no valid angular sampling")
                    wl_min = float(
                        product.provenance.get("wl_min_um", item["header"].get("WL_MIN"))
                    )
                    wl_max = float(
                        product.provenance.get("wl_max_um", item["header"].get("WL_MAX"))
                    )
                    positions = _sample_positions(group["bounds"])
                    nhits = _load_nhits(product)
                    if group["snapshot"]["selection"] == "candidate" and nhits is None:
                        raise ValueError("queried exposure selection requires mosaic NHITS validation")
                    samples = []
                    sample_qc = []
                    sample_failed = False
                    for sample_index, position in enumerate(positions):
                        try:
                            model, metrics = _synthesize_sample(
                                exposures,
                                item["header"],
                                output_sampling,
                                position,
                                wl_min,
                                wl_max,
                                nhits,
                                f"spherex-mosaic:{product.product_id}:{position}",
                                sample_index=sample_index,
                            )
                        except Exception as exc:
                            failures[
                                f"{key}/sample-{sample_index}:{position[0]:.3f},{position[1]:.3f}"
                            ] = str(exc)
                            sample_failed = True
                            continue
                        samples.append(PSFSample(
                            model,
                            position=position,
                            weight=1.0,
                            metadata=metrics,
                        ))
                        sample_qc.append({"position": position, **metrics})
                    if sample_failed:
                        continue
                    ensemble = PSFEnsemble(samples, provenance={
                        "provider": "spherex-mosaic-synthesis",
                        "product_group_id": group_id,
                        "exposure_snapshot_digest": _stable_digest(group["snapshot"]),
                        "recipe_digest": _stable_digest(recipe),
                    })
                    ensemble.spatial_metrics = summarize_spatial_ensemble(
                        ensemble,
                        r80_tolerance=_SPATIAL_R80_TOLERANCE,
                        ee_tolerance=_SPATIAL_EE_TOLERANCE,
                        l1_tolerance=_SPATIAL_L1_TOLERANCE,
                        centroid_tolerance=_SPATIAL_CENTROID_TOLERANCE,
                    )
                    representative = representative_from_ensemble(
                        ensemble,
                        f"spherex-mosaic-ensemble:{band}",
                        validator=validate_effective_psf,
                        edge_tolerance=_PSF_EDGE_TOLERANCE,
                    )
                    filename = f"{product.product_id}--{band}.fits"
                    _write_psf_artifact(temporary / filename, representative, samples)
                    artifact_names.append(filename)
                    sampling = output_sampling.to_dict()
                    manifest_models.append({
                        "selector": {
                            "product_id": product.product_id,
                            "facility": "SPHEREx",
                            "channel": str(band),
                            "valid_region": list(group["bounds"]),
                        },
                        "representation": {
                            "path": filename,
                            "hdu": 0,
                            "sampling": sampling,
                            "pixel_integrated": True,
                        },
                        "applicability": "output_product",
                        "provenance": {
                            "release": product.release,
                            "calibration_id": "qr3-epsf",
                            "product_group_id": group_id,
                            "exposure_snapshot_digest": _stable_digest(group["snapshot"]),
                            "recipe_digest": _stable_digest(recipe),
                            "build_digest": build_digest,
                        },
                        "spatial_validation": {
                            "samples": [
                                {
                                    "path": filename,
                                    "hdu": index,
                                    "position": list(sample.position),
                                    "weight": sample.weight,
                                }
                                for index, sample in enumerate(samples, start=1)
                            ],
                            "metrics": ensemble.spatial_metrics,
                        },
                    })
                    group_qc["bands"][str(band)] = {
                        "samples": sample_qc,
                        "spatial": ensemble.spatial_metrics,
                        "representative_fwhm_arcsec": representative.fwhm_arcsec,
                        "representative_r80_arcsec": representative.r80_arcsec,
                    }
                except Exception as exc:
                    failures[key] = str(exc)
        if failures:
            raise SpherexMosaicPSFBuildError(failures)
        manifest = {
            "schema_version": 2,
            "generated_by": "Spec7DT.spherex_mosaic_psf",
            "build_digest": build_digest,
            "models": manifest_models,
        }
        _write_json_atomic(temporary / "psf_manifest.json", manifest)
        _write_json_atomic(temporary / "qc.json", qc)
        if final_dir.exists():
            shutil.rmtree(temporary)
        else:
            try:
                temporary.replace(final_dir)
            except FileExistsError:
                shutil.rmtree(temporary)
        final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_qc = json.loads(qc_path.read_text(encoding="utf-8"))
        return SpherexMosaicPSFBuildResult(
            final_manifest,
            manifest_path,
            snapshot_path,
            tuple(final_dir / name for name in sorted(artifact_names)),
            final_qc,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "SpherexMosaicPSFBuildError",
    "SpherexMosaicPSFBuildResult",
    "SpherexMosaicPSFSynthesisConfig",
    "build_spherex_mosaic_psf_manifest",
    "merge_psf_manifests",
    "spherex_mosaic_synthesis_required",
    "synthesis_config_for_analysis_footprint",
]
