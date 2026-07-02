from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


class GalaxyMetadataError(RuntimeError):
    """Raised when required galaxy metadata cannot be resolved safely."""

    def __init__(self, galaxy_name: str, metadata_type: str, tried_sources: list[str]):
        self.galaxy_name = galaxy_name
        self.metadata_type = metadata_type
        self.tried_sources = tried_sources
        sources = ", ".join(tried_sources) if tried_sources else "none"
        super().__init__(
            f"Could not resolve {metadata_type} for '{galaxy_name}'. "
            f"Tried: {sources}. Provide galaxy_metadata={{'{galaxy_name}': "
            "{'coord': (ra_deg, dec_deg), 'redshift': z}}} or configure a "
            "local metadata cache."
        )


@dataclass
class GalaxyMetadataConfig:
    use_remote: bool = True
    remote_sources: tuple[str, ...] = ("ned", "simbad")
    fallback_order: tuple[str, ...] = ("remote", "manual", "header", "cache")
    timeout: float = 10.0
    retries: int = 2
    cache: bool = True
    cache_path: str | Path | None = None
    warn: bool = True

    def __post_init__(self):
        self.remote_sources = tuple(str(source).lower() for source in self.remote_sources)
        self.fallback_order = tuple(str(source).lower() for source in self.fallback_order)

    @property
    def resolved_cache_path(self) -> Path:
        if self.cache_path is not None:
            return Path(self.cache_path).expanduser()
        env_path = os.environ.get("SPEC7DT_METADATA_CACHE")
        if env_path:
            return Path(env_path).expanduser()
        return Path("~/.cache/Spec7DT/galaxy_metadata.json").expanduser()


class GalaxyMetadataResolver:
    """Resolve galaxy coordinates and redshifts with configurable fallbacks."""

    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
        config: GalaxyMetadataConfig | dict[str, Any] | None = None,
        ned_client: Any | None = None,
        simbad_client: Any | None = None,
    ):
        if isinstance(config, dict):
            config = GalaxyMetadataConfig(**config)
        self.config = config or GalaxyMetadataConfig()
        self.manual = self._normalize_metadata(metadata or {})
        self._ned_client = ned_client
        self._simbad_client = simbad_client
        self._cache = self._load_cache()
        self._remote_results: dict[tuple[str, str], dict[str, Any]] = {}

    def get_coord(self, galaxy_name: str, header: Any | None = None, required: bool = True) -> tuple[float, float] | None:
        tried = []
        for source in self.config.fallback_order:
            if source == "remote":
                coord, remote_tried = self._remote_value(galaxy_name, "coord")
                tried.extend(remote_tried)
                if coord is not None:
                    return coord
            elif source == "manual":
                tried.append("manual")
                coord = self._record_coord(self.manual.get(galaxy_name))
                if coord is not None:
                    return coord
            elif source == "header":
                if header is None:
                    continue
                tried.append("header")
                coord = self._coord_from_header(header)
                if coord is not None:
                    return coord
            elif source == "cache":
                if not self.config.cache:
                    continue
                tried.append("cache")
                coord = self._record_coord(self._cache.get(galaxy_name))
                if coord is not None:
                    return coord
            else:
                tried.append(f"{source} ignored")

        if required:
            raise GalaxyMetadataError(galaxy_name, "coordinate", tried)
        self._warn(f"Could not resolve coordinate for {galaxy_name}; continuing without it.")
        return None

    def get_skycoord(self, galaxy_name: str, header: Any | None = None, required: bool = True) -> SkyCoord | None:
        coord = self.get_coord(galaxy_name, header=header, required=required)
        if coord is None:
            return None
        return SkyCoord(ra=coord[0] * u.deg, dec=coord[1] * u.deg, frame="icrs")

    def get_redshift(self, galaxy_name: str) -> float | None:
        for source in self.config.fallback_order:
            if source == "remote":
                redshift, _ = self._remote_value(galaxy_name, "redshift")
                if redshift is not None:
                    return redshift
            elif source == "manual":
                redshift = self._record_redshift(self.manual.get(galaxy_name))
                if redshift is not None:
                    return redshift
            elif source == "cache":
                if not self.config.cache:
                    continue
                redshift = self._record_redshift(self._cache.get(galaxy_name))
                if redshift is not None:
                    return redshift

        self._warn(f"Could not resolve redshift for {galaxy_name}; using NaN.")
        return None

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.config.cache:
            return {}
        path = self.config.resolved_cache_path
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            self._warn(f"Could not read metadata cache {path}: {exc}")
            return {}
        if not isinstance(data, dict):
            self._warn(f"Metadata cache {path} does not contain a JSON object.")
            return {}
        return self._normalize_metadata(data)

    def _save_cache(self) -> None:
        if not self.config.cache:
            return
        path = self.config.resolved_cache_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as file:
                json.dump(self._cache, file, indent=2, sort_keys=True)
        except Exception as exc:
            self._warn(f"Could not write metadata cache {path}: {exc}")

    def _remote_value(self, galaxy_name: str, value_type: str) -> tuple[Any | None, list[str]]:
        if not self.config.use_remote:
            return None, ["remote disabled"]

        tried = []
        for source in self.config.remote_sources:
            source = source.lower()
            tried.append(source.upper())
            record = self._query_remote_source(galaxy_name, source)
            if value_type == "coord":
                value = self._record_coord(record)
            elif value_type == "redshift":
                value = self._record_redshift(record)
            else:
                value = None
            if value is not None:
                return value, tried
        return None, tried

    def _query_remote_source(self, galaxy_name: str, source: str) -> dict[str, Any]:
        cache_key = (source, galaxy_name)
        if cache_key in self._remote_results:
            return self._remote_results[cache_key]

        result: dict[str, Any] = {}
        last_error: Exception | None = None
        for _ in range(max(1, int(self.config.retries) + 1)):
            try:
                if source == "ned":
                    result = self._query_ned(galaxy_name)
                elif source == "simbad":
                    result = self._query_simbad(galaxy_name)
                else:
                    self._warn(f"Unknown remote metadata source '{source}' for {galaxy_name}.")
                    result = {}
                break
            except Exception as exc:
                last_error = exc

        if not result:
            if last_error is not None:
                self._warn(f"{source.upper()} query failed for {galaxy_name}: {last_error}")
            self._remote_results[cache_key] = {}
            return {}

        self._remote_results[cache_key] = result
        if self.config.cache:
            cached = self._cache.setdefault(galaxy_name, {})
            cached.update(result)
            self._save_cache()
        return result

    def _query_ned(self, galaxy_name: str) -> dict[str, Any]:
        client = self._get_ned_client()
        self._set_client_timeout(client)
        table = client.query_object(galaxy_name)
        return self._metadata_from_ned_table(table)

    def _query_simbad(self, galaxy_name: str) -> dict[str, Any]:
        client = self._get_simbad_client()
        self._set_client_timeout(client)
        self._configure_simbad_fields(client)
        table = client.query_object(galaxy_name)
        return self._metadata_from_simbad_table(table)

    def _get_ned_client(self):
        if self._ned_client is not None:
            return self._ned_client
        from astroquery.ipac.ned import Ned

        self._ned_client = Ned
        return self._ned_client

    def _get_simbad_client(self):
        if self._simbad_client is not None:
            return self._simbad_client
        from astroquery.simbad import Simbad

        try:
            self._simbad_client = Simbad()
        except Exception:
            self._simbad_client = Simbad
        return self._simbad_client

    def _configure_simbad_fields(self, client: Any) -> None:
        try:
            client.add_votable_fields("rvz_redshift")
        except Exception:
            pass

    def _set_client_timeout(self, client: Any) -> None:
        try:
            setattr(client, "TIMEOUT", self.config.timeout)
        except Exception:
            pass

    def _metadata_from_ned_table(self, table: Any) -> dict[str, Any]:
        try:
            if len(table) == 0:
                return {}
        except Exception:
            return {}

        result: dict[str, Any] = {}
        try:
            coord = self._coord_from_value((table["RA"][0], table["DEC"][0]))
            if coord is not None:
                result["coord"] = [coord[0], coord[1]]
        except Exception:
            pass

        try:
            redshift = self._redshift_from_value(table["Redshift"][0])
            if redshift is not None:
                result["redshift"] = redshift
        except Exception:
            pass

        return result

    def _metadata_from_simbad_table(self, table: Any) -> dict[str, Any]:
        try:
            if len(table) == 0:
                return {}
        except Exception:
            return {}

        result: dict[str, Any] = {}
        names = self._table_colnames(table)
        ra_key = self._first_matching_column(names, "RA", "ra")
        dec_key = self._first_matching_column(names, "DEC", "dec")
        if ra_key is not None and dec_key is not None:
            try:
                coord = self._coord_from_simbad_value((table[ra_key][0], table[dec_key][0]))
                if coord is not None:
                    result["coord"] = [coord[0], coord[1]]
            except Exception:
                pass

        for redshift_key in ("RVZ_REDSHIFT", "rvz_redshift", "Z_VALUE", "z_value", "REDSHIFT", "redshift", "Redshift", "z"):
            column = self._first_matching_column(names, redshift_key)
            if column is None:
                continue
            try:
                redshift = self._redshift_from_value(table[column][0])
            except Exception:
                redshift = None
            if redshift is not None:
                result["redshift"] = redshift
                break

        return result

    def _table_colnames(self, table: Any) -> set[str]:
        names = getattr(table, "colnames", None)
        if names is not None:
            return set(names)
        if isinstance(table, dict):
            return set(table.keys())
        return set()

    def _first_matching_column(self, names: set[str], *candidates: str) -> str | None:
        lower_names = {name.lower(): name for name in names}
        for candidate in candidates:
            if candidate in names:
                return candidate
            match = lower_names.get(candidate.lower())
            if match is not None:
                return match
        return None

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for galaxy_name, record in metadata.items():
            if record is None:
                continue
            normalized_record: dict[str, Any] = {}

            if isinstance(record, SkyCoord) or (
                isinstance(record, (list, tuple, np.ndarray)) and len(record) >= 2
            ):
                coord_value = record
                redshift_value = None
            elif isinstance(record, dict):
                coord_value = record.get("coord", record.get("coordinate", record.get("skycoord")))
                if coord_value is None and "ra" in record and "dec" in record:
                    coord_value = (record["ra"], record["dec"])
                redshift_value = record.get("redshift", record.get("z"))
            else:
                continue

            coord = self._coord_from_value(coord_value)
            if coord is not None:
                normalized_record["coord"] = [coord[0], coord[1]]

            redshift = self._redshift_from_value(redshift_value)
            if redshift is not None:
                normalized_record["redshift"] = redshift

            if normalized_record:
                normalized[str(galaxy_name)] = normalized_record
        return normalized

    def _record_coord(self, record: dict[str, Any] | None) -> tuple[float, float] | None:
        if not record:
            return None
        return self._coord_from_value(record.get("coord"))

    def _record_redshift(self, record: dict[str, Any] | None) -> float | None:
        if not record:
            return None
        return self._redshift_from_value(record.get("redshift"))

    def _coord_from_header(self, header: Any) -> tuple[float, float] | None:
        for ra_key, dec_key in (("RA", "DEC"), ("OBJRA", "OBJDEC")):
            try:
                coord = self._coord_from_value((header[ra_key], header[dec_key]))
            except Exception:
                coord = None
            if coord is not None:
                return coord
        return None

    def _coord_from_simbad_value(self, value: Any) -> tuple[float, float] | None:
        if value is None or not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
            return self._coord_from_value(value)
        try:
            coord = SkyCoord(value[0], value[1], unit=(u.hourangle, u.deg), frame="icrs")
            return float(coord.ra.deg), float(coord.dec.deg)
        except Exception:
            return self._coord_from_value(value)

    def _coord_from_value(self, value: Any) -> tuple[float, float] | None:
        if value is None:
            return None
        if isinstance(value, SkyCoord):
            return float(value.ra.deg), float(value.dec.deg)
        if isinstance(value, dict):
            if "ra" in value and "dec" in value:
                return self._coord_from_value((value["ra"], value["dec"]))
            return None
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
            return None

        ra_value, dec_value = value[0], value[1]
        try:
            ra = float(ra_value)
            dec = float(dec_value)
            if np.isfinite(ra) and np.isfinite(dec):
                return ra, dec
        except Exception:
            pass

        for unit in ((u.deg, u.deg), (u.hourangle, u.deg)):
            try:
                coord = SkyCoord(ra_value, dec_value, unit=unit, frame="icrs")
                return float(coord.ra.deg), float(coord.dec.deg)
            except Exception:
                continue
        return None

    def _redshift_from_value(self, value: Any) -> float | None:
        if value is None or np.ma.is_masked(value):
            return None
        try:
            redshift = float(value)
        except Exception:
            return None
        if np.isnan(redshift):
            return None
        return redshift

    def _warn(self, message: str) -> None:
        if self.config.warn:
            warnings.warn(message, RuntimeWarning)
