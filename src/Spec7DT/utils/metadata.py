from __future__ import annotations

import json
import os
import warnings
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse_single_table
from astropy.wcs import WCS


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
            "{'coord': (ra_deg, dec_deg), 'redshift': z, 'distance': mpc}}} "
            "or configure a "
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
    """Resolve galaxy coordinates, redshifts, and distances with fallbacks."""

    _NED_OVERVIEW_URL = "https://ned.ipac.caltech.edu/NED::API/OverviewOfObject"
    _CACHE_SCHEMA_KEY = "__schema_version__"
    _CACHE_SCHEMA_VERSION = 3

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
        self._last_coord_source: dict[str, str] = {}

    def get_coord(
        self,
        galaxy_name: str,
        header: Any | None = None,
        required: bool = True,
        image_shape: tuple[int, int] | None = None,
    ) -> tuple[float, float] | None:
        tried = []
        for source in self.config.fallback_order:
            if source == "remote":
                if not self.config.use_remote:
                    tried.append("remote disabled")
                    continue
                for remote_source in self.config.remote_sources:
                    label = remote_source.upper()
                    tried.append(label)
                    record = self._query_remote_source(galaxy_name, remote_source.lower())
                    coord = self._record_coord(record)
                    if self._accept_coord(
                        galaxy_name,
                        label,
                        coord,
                        header=header,
                        image_shape=image_shape,
                    ):
                        self._cache_remote_record(galaxy_name, record, coord=coord)
                        self._last_coord_source[galaxy_name] = label
                        return coord
                    self._cache_remote_record(galaxy_name, record)
            elif source == "manual":
                tried.append("manual")
                coord = self._record_coord(self.manual.get(galaxy_name))
                if self._accept_coord(
                    galaxy_name,
                    "manual",
                    coord,
                    header=header,
                    image_shape=image_shape,
                ):
                    self._last_coord_source[galaxy_name] = "manual"
                    return coord
            elif source == "header":
                if header is None:
                    continue
                tried.append("header")
                coord = self._coord_from_header(header)
                if self._accept_coord(
                    galaxy_name,
                    "header",
                    coord,
                    header=header,
                    image_shape=image_shape,
                ):
                    self._last_coord_source[galaxy_name] = "header"
                    return coord
            elif source == "cache":
                if not self.config.cache:
                    continue
                tried.append("cache")
                coord = self._record_coord(self._cache.get(galaxy_name))
                if self._accept_coord(
                    galaxy_name,
                    "cache",
                    coord,
                    header=header,
                    image_shape=image_shape,
                ):
                    self._last_coord_source[galaxy_name] = "cache"
                    return coord
            else:
                tried.append(f"{source} ignored")

        if required:
            raise GalaxyMetadataError(galaxy_name, "coordinate", tried)
        self._warn(f"Could not resolve coordinate for {galaxy_name}; continuing without it.")
        return None

    def get_skycoord(
        self,
        galaxy_name: str,
        header: Any | None = None,
        required: bool = True,
        image_shape: tuple[int, int] | None = None,
    ) -> SkyCoord | None:
        coord = self.get_coord(
            galaxy_name,
            header=header,
            required=required,
            image_shape=image_shape,
        )
        if coord is None:
            return None
        return SkyCoord(ra=coord[0] * u.deg, dec=coord[1] * u.deg, frame="icrs")

    def get_coord_source(self, galaxy_name: str) -> str | None:
        """Return the source used by the most recent coordinate resolution."""
        return self._last_coord_source.get(galaxy_name)

    def _accept_coord(
        self,
        galaxy_name: str,
        source: str,
        coord: tuple[float, float] | None,
        *,
        header: Any | None,
        image_shape: tuple[int, int] | None,
    ) -> bool:
        if coord is None:
            return False

        ra, dec = coord
        if not (np.isfinite(ra) and np.isfinite(dec) and 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            self._warn(
                f"Rejected {source} coordinate for {galaxy_name}: {coord} is not a valid ICRS coordinate."
            )
            return False

        if image_shape is None:
            return True
        if header is None:
            self._warn(
                f"Rejected {source} coordinate for {galaxy_name}: image bounds were supplied without a FITS header."
            )
            return False

        try:
            skycoord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            x, y = WCS(header).world_to_pixel(skycoord)
            height, width = tuple(image_shape)[-2:]
            inside = (
                np.isfinite(x)
                and np.isfinite(y)
                and 0.0 <= float(x) < float(width)
                and 0.0 <= float(y) < float(height)
            )
        except Exception as exc:
            self._warn(
                f"Rejected {source} coordinate for {galaxy_name}: {coord} could not be projected "
                f"into the image ({exc})."
            )
            return False

        if not inside:
            self._warn(
                f"Rejected {source} coordinate for {galaxy_name}: {coord} projects to "
                f"({x}, {y}), outside image shape {tuple(image_shape)}."
            )
            return False
        return True

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

    def get_distance(self, galaxy_name: str) -> float | None:
        """Return the preferred NED distance in Mpc, if available."""
        for source in self.config.fallback_order:
            if source == "remote":
                distance, _ = self._remote_value(galaxy_name, "distance")
                if distance is not None:
                    return distance
            elif source == "manual":
                distance = self._record_distance(self.manual.get(galaxy_name))
                if distance is not None:
                    return distance
            elif source == "cache":
                if not self.config.cache:
                    continue
                distance = self._record_distance(self._cache.get(galaxy_name))
                if distance is not None:
                    return distance

        self._warn(f"Could not resolve distance for {galaxy_name}; using NaN.")
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
        cache_version = data.pop(self._CACHE_SCHEMA_KEY, None)
        normalized = self._normalize_metadata(data)
        if cache_version != self._CACHE_SCHEMA_VERSION:
            for record in normalized.values():
                record.pop("coord", None)
            self._warn(
                "Ignoring coordinates from a pre-v3 metadata cache because older SIMBAD "
                "responses may have been interpreted with the wrong RA unit; redshift and "
                "distance values were preserved."
            )
        return normalized

    def _save_cache(self) -> None:
        if not self.config.cache:
            return
        path = self.config.resolved_cache_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {self._CACHE_SCHEMA_KEY: self._CACHE_SCHEMA_VERSION, **self._cache}
            with path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
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
            elif value_type == "distance":
                value = self._record_distance(record)
            else:
                value = None
            if value is not None:
                self._cache_remote_value(galaxy_name, value_type, value)
                return value, tried
        return None, tried

    def _cache_remote_record(
        self,
        galaxy_name: str,
        record: dict[str, Any],
        *,
        coord: tuple[float, float] | None = None,
    ) -> None:
        """Cache independently validated fields from a coordinate lookup."""
        if not self.config.cache or not record:
            return

        validated: dict[str, Any] = {}
        if coord is not None:
            validated["coord"] = [float(coord[0]), float(coord[1])]
        redshift = self._record_redshift(record)
        if redshift is not None:
            validated["redshift"] = redshift
        distance = self._record_distance(record)
        if distance is not None:
            validated["distance"] = distance
        self._update_cache(galaxy_name, validated)

    def _cache_remote_value(self, galaxy_name: str, value_type: str, value: Any) -> None:
        """Cache only the scalar field requested by a scalar metadata lookup."""
        if value_type not in {"redshift", "distance"}:
            return
        self._update_cache(galaxy_name, {value_type: value})

    def _update_cache(self, galaxy_name: str, values: dict[str, Any]) -> None:
        if not self.config.cache or not values:
            return
        cached = self._cache.setdefault(galaxy_name, {})
        if all(cached.get(key) == value for key, value in values.items()):
            return
        cached.update(values)
        self._save_cache()

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
        return result

    def _query_ned(self, galaxy_name: str) -> dict[str, Any]:
        if self._ned_client is not None:
            client = self._get_ned_client()
            self._set_client_timeout(client)
            table = client.query_object(galaxy_name)
        else:
            table = self._query_ned_overview(galaxy_name)
        return self._metadata_from_ned_table(table)

    def _query_ned_overview(self, galaxy_name: str):
        query = urlencode({"TARGET": galaxy_name})
        url = f"{self._NED_OVERVIEW_URL}?{query}"
        with urlopen(url, timeout=self.config.timeout) as response:
            payload = response.read()
        return parse_single_table(BytesIO(payload)).to_table(use_names_over_ids=True)

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
        coord = self._ned_equatorial_coord(table)
        if coord is not None:
            result["coord"] = [coord[0], coord[1]]

        redshift = self._redshift_from_value(
            self._first_table_value(table, "Redshift", "Redshift (z)")
        )
        if redshift is not None:
            result["redshift"] = redshift

        mean_distance = self._distance_from_value(
            self._first_table_value(table, "Mean Distance", "Mean_Distance")
        )
        cmb_distance = self._distance_from_value(
            self._first_table_value(table, "D (3K CMB)", "D_3K_CMB")
        )
        distance = mean_distance if mean_distance is not None else cmb_distance
        if distance is not None:
            result["distance"] = distance

        return result

    def _ned_equatorial_coord(self, table: Any) -> tuple[float, float] | None:
        """Read NED's decimal-degree coordinate pair before sexagesimal fields."""
        ra_column = self._ned_degree_column(table, axis="ra")
        dec_column = self._ned_degree_column(table, axis="dec")
        if ra_column is not None and dec_column is not None:
            try:
                coord = self._coord_from_value((table[ra_column][0], table[dec_column][0]))
            except Exception:
                coord = None
            if coord is not None:
                return coord

        names = self._table_colnames(table)
        ra_sex = self._first_matching_column(
            names,
            "Lon (Equatorial J2000) in sexagesimal",
            "RA-s",
        )
        dec_sex = self._first_matching_column(
            names,
            "Lat (Equatorial J2000) in sexagesimal",
            "Dec-s",
        )
        if ra_sex is None or dec_sex is None:
            return None
        try:
            coord = SkyCoord(
                table[ra_sex][0],
                table[dec_sex][0],
                unit=(u.hourangle, u.deg),
                frame="icrs",
            )
        except Exception:
            return None
        return float(coord.ra.deg), float(coord.dec.deg)

    def _ned_degree_column(self, table: Any, axis: str) -> str | None:
        names = sorted(self._table_colnames(table))
        scored: list[tuple[int, str]] = []
        for name in names:
            normalized = self._normalize_column_name(name)
            is_ra = normalized == "ra" or (
                normalized.startswith("lonequatorialj2000") and "sexagesimal" not in normalized
            )
            is_dec = normalized == "dec" or (
                normalized.startswith("latequatorialj2000") and "sexagesimal" not in normalized
            )
            if (axis == "ra" and not is_ra) or (axis == "dec" and not is_dec):
                continue

            try:
                column = table[name]
                value = column[0]
            except Exception:
                continue
            try:
                numeric = np.issubdtype(np.asarray(value).dtype, np.number)
            except TypeError:
                numeric = False
            unit = getattr(column, "unit", None)
            unit_is_degree = unit is not None and str(unit).lower() in {"deg", "degree"}
            if not numeric:
                continue
            score = (100 if unit_is_degree else 0) + (20 if "equatorialj2000" in normalized else 0)
            scored.append((score, name))

        if not scored:
            return None
        return max(scored)[1]

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
                coord = self._coord_from_simbad_columns(table, ra_key, dec_key)
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

    def _coord_from_simbad_columns(
        self,
        table: Any,
        ra_key: str,
        dec_key: str,
    ) -> tuple[float, float] | None:
        ra_column = table[ra_key]
        dec_column = table[dec_key]
        ra_value = ra_column[0]
        dec_value = dec_column[0]

        if self._is_numeric_coord_pair(ra_value, dec_value):
            ra = self._numeric_angle_to_degree(
                ra_value,
                getattr(ra_column, "unit", None),
                default_unit=u.deg,
            )
            dec = self._numeric_angle_to_degree(
                dec_value,
                getattr(dec_column, "unit", None),
                default_unit=u.deg,
            )
            return self._coord_from_value((ra, dec))

        return self._coord_from_simbad_value((ra_value, dec_value))

    @staticmethod
    def _is_numeric_coord_pair(ra_value: Any, dec_value: Any) -> bool:
        if np.ma.is_masked(ra_value) or np.ma.is_masked(dec_value):
            return False
        try:
            return np.isfinite(float(ra_value)) and np.isfinite(float(dec_value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _numeric_angle_to_degree(value: Any, unit: Any, *, default_unit: u.Unit) -> float:
        angle_unit = default_unit if unit is None else u.Unit(unit)
        return float((float(value) * angle_unit).to_value(u.deg))

    def _table_colnames(self, table: Any) -> set[str]:
        names = getattr(table, "colnames", None)
        if names is not None:
            return set(names)
        if isinstance(table, dict):
            return set(table.keys())
        return set()

    def _first_matching_column(self, names: set[str], *candidates: str) -> str | None:
        lower_names = {name.lower(): name for name in names}
        normalized_names = {self._normalize_column_name(name): name for name in names}
        for candidate in candidates:
            if candidate in names:
                return candidate
            match = lower_names.get(candidate.lower())
            if match is not None:
                return match
            match = normalized_names.get(self._normalize_column_name(candidate))
            if match is not None:
                return match
        return None

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        return "".join(character.lower() for character in str(name) if character.isalnum())

    def _first_table_value(self, table: Any, *candidates: str) -> Any | None:
        column = self._first_matching_column(self._table_colnames(table), *candidates)
        if column is None:
            return None
        try:
            return table[column][0]
        except Exception:
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
                distance_value = record.get("distance", record.get("distance_mpc"))
            else:
                continue

            if not isinstance(record, dict):
                distance_value = None

            coord = self._coord_from_value(coord_value)
            if coord is not None:
                normalized_record["coord"] = [coord[0], coord[1]]

            redshift = self._redshift_from_value(redshift_value)
            if redshift is not None:
                normalized_record["redshift"] = redshift

            distance = self._distance_from_value(distance_value)
            if distance is not None:
                normalized_record["distance"] = distance

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

    def _record_distance(self, record: dict[str, Any] | None) -> float | None:
        if not record:
            return None
        return self._distance_from_value(record.get("distance"))

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
        if self._is_numeric_coord_pair(value[0], value[1]):
            return self._coord_from_value(value)
        try:
            coord = SkyCoord(value[0], value[1], unit=(u.hourangle, u.deg), frame="icrs")
            return float(coord.ra.deg), float(coord.dec.deg)
        except Exception:
            return None

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
            if (
                np.isfinite(ra)
                and np.isfinite(dec)
                and 0.0 <= ra < 360.0
                and -90.0 <= dec <= 90.0
            ):
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
        if not np.isfinite(redshift):
            return None
        return redshift

    def _distance_from_value(self, value: Any) -> float | None:
        if value is None or np.ma.is_masked(value):
            return None
        try:
            distance = float(value)
        except Exception:
            return None
        if not np.isfinite(distance) or distance <= 0:
            return None
        return distance

    def _warn(self, message: str) -> None:
        if self.config.warn:
            warnings.warn(message, RuntimeWarning)
