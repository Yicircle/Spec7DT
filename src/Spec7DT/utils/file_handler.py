import numpy as np
from pathlib import Path
from astropy.io import fits
import re
import inspect
import warnings
from glob import glob
from rich.console import Console
from rich.tree import Tree

from .utility import Observatories
from .utility import useful_functions
from .file_generator import *
from .reporting import emit_alert
from ..plot.plot import DrawGalaxy
from ..handlers.filter_handler import Filters
from ..core import CatalogFrame


class DuplicateImageError(ValueError):
    """Raised when multiple files map to the same image identity."""

    def __init__(self, collisions):
        normalized = {}
        for key, paths in collisions.items():
            normalized[tuple(key)] = tuple(sorted({str(path) for path in paths}))
        self.collisions = dict(sorted(normalized.items()))
        details = []
        for (galaxy, observatory, band), paths in self.collisions.items():
            details.append(
                f"  {galaxy}/{observatory}/{band}: " + ", ".join(paths)
            )
        super().__init__(
            "Duplicate image keys were found; each galaxy/observatory/band must map "
            "to exactly one input:\n" + "\n".join(details)
        )


class GalaxyImageSet:
    __signature__ = inspect.Signature()
    _IMAGE_EXTENSION_KEYS = ("EXTNAME", "EXTTYPE", "HDUCLAS1", "HDUCLAS2")
    _IMAGE_EXTENSION_VALUES = {"IMAGE", "SCI", "SCIENCE", "DATA"}
    _HEADER_MERGE_SKIP_KEYS = {
        "",
        "XTENSION",
        "PCOUNT",
        "GCOUNT",
        "CHECKSUM",
        "DATASUM",
        "ZIMAGE",
        "ZBITPIX",
        "ZNAXIS",
        "ZCMPTYPE",
        "ZQUANTIZ",
        "ZDITHER0",
        "ZBLANK",
    }
    _HEADER_MERGE_SKIP_PREFIXES = ("ZTILE", "ZNAME", "ZVAL")

    def __init__(
        self,
        auto_ensure_filters=True,
        unknown_policy="best_effort",
        filter_config=None,
    ):
        if unknown_policy not in {"best_effort", "strict"}:
            raise ValueError("unknown_policy must be 'best_effort' or 'strict'")

        self._data = {}
        self._header = {}
        self._error = {}
        self._error_source = {}
        self._psf = {}
        self._cutout_shape = {}
        self._obs = {}
        self._files = {}
        self._last_catalog = None
        self.filter_inst = Filters()
        self.auto_ensure_filters = bool(auto_ensure_filters)
        self.unknown_policy = unknown_policy
        self.filter_config = {"allow_svo": True, "cache": True, "warn": True}
        if filter_config:
            self.filter_config.update(filter_config)

    def _invalidate_last_catalog(self):
        self._last_catalog = None

    def _error_sources(self):
        """Return the provenance tree, initializing legacy unpickled instances."""
        if not hasattr(self, "_error_source"):
            self._error_source = {}
        return self._error_source

    def _set_last_catalog(self, catalog_frame):
        if not isinstance(catalog_frame, CatalogFrame):
            raise TypeError("catalog_frame must be a CatalogFrame instance")
        self._last_catalog = catalog_frame

    def _require_last_catalog(self):
        if self._last_catalog is None:
            raise RuntimeError(
                "No pipeline catalog is available. Run execute_pipeline() before "
                "plot_sed() or write_catalog()."
            )
        return self._last_catalog

    @property
    def last_catalog(self):
        """Most recent pipeline catalog, or ``None`` after image mutation."""
        return self._last_catalog

    def add_image(self, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        image_data, image_header = self._read_fits_image(filepath)

        
        file_name = filepath.stem
        galaxy_name = Parsers._galaxy_name_parser(file_name=file_name)
        best_effort = self.unknown_policy == "best_effort"
        observatory = Parsers._observatory_name_parser(
            file_name=file_name,
            header=image_header,
            best_effort=best_effort,
        )
        band = Parsers._band_name_parser(
            file_name=file_name,
            filter_inst=self.filter_inst,
            header=image_header,
            observatory=observatory,
            best_effort=best_effort,
        )
        if galaxy_name is None or band is None or observatory is None:
            warnings.warn(
                f"Skipping {filepath}: could not parse galaxy, observatory, and band."
            )
            return

        existing_path = (
            self._files.get(galaxy_name, {})
            .get(observatory, {})
            .get(band)
        )
        if band in self._data.get(galaxy_name, {}).get(observatory, {}):
            key = (galaxy_name, observatory, band)
            raise DuplicateImageError(
                {key: (existing_path or f"<in-memory:{'/'.join(key)}>", filepath)}
            )

        if self.auto_ensure_filters:
            self._ensure_filter_for_image(filepath, observatory, band)

        error_sources = self._error_sources()
        for attr in [
            self._data,
            self._header,
            self._error,
            error_sources,
            self._psf,
            self._cutout_shape,
            self._files,
        ]:
            attr.setdefault(galaxy_name, {}).setdefault(observatory, {})

        self._data[galaxy_name][observatory][band] = image_data
        self._header[galaxy_name][observatory][band] = image_header
        self._psf[galaxy_name][observatory][band] = ""
        self._cutout_shape[galaxy_name][observatory][band] = set()
        self._files[galaxy_name][observatory][band] = str(filepath)
        self._invalidate_last_catalog()

        err_name = filepath.with_name(filepath.stem + "_err").with_suffix(filepath.suffix)
        if not err_name.exists():
            warnings.warn(
                f"Error File {err_name} not found. Marking the error map as missing; "
                "execute_pipeline() will estimate it from the background RMS."
            )
            self._error[galaxy_name][observatory][band] = None
            self._error_source[galaxy_name][observatory][band] = "missing"
        else:
            error_data, _ = self._read_fits_image(err_name)
            self._error[galaxy_name][observatory][band] = error_data
            self._error_source[galaxy_name][observatory][band] = "file"

    @classmethod
    def _read_fits_image(cls, filepath):
        with fits.open(filepath) as hdul:
            _, image_hdu = cls._find_image_hdu(hdul, filepath)
            image_data = np.array(image_hdu.data, dtype=np.float32)
            image_header = cls._merge_primary_and_image_headers(hdul[0].header, image_hdu.header)
        return image_data, image_header

    @classmethod
    def _find_image_hdu(cls, hdul, filepath):
        image_candidates = []
        marked_candidates = []

        for index, hdu in enumerate(hdul):
            if not cls._is_2d_image_hdu(hdu):
                continue

            image_candidates.append((index, hdu))
            if cls._has_image_extension_marker(hdu.header):
                marked_candidates.append((index, hdu))

        if marked_candidates:
            return marked_candidates[0]
        if image_candidates:
            return image_candidates[0]

        raise ValueError(f"No 2D image HDU found in {filepath}")

    @staticmethod
    def _is_2d_image_hdu(hdu):
        data = getattr(hdu, "data", None)
        shape = getattr(data, "shape", None)
        return data is not None and shape is not None and len(shape) == 2 and np.prod(shape) > 0

    @classmethod
    def _has_image_extension_marker(cls, header):
        for key in cls._IMAGE_EXTENSION_KEYS:
            value = header.get(key)
            if value is None:
                continue
            if str(value).strip().upper() in cls._IMAGE_EXTENSION_VALUES:
                return True
        return False

    @classmethod
    def _merge_primary_and_image_headers(cls, primary_header, image_header):
        merged_header = primary_header.copy()
        if image_header is primary_header:
            return merged_header

        for card in image_header.cards:
            key = card.keyword
            if cls._should_skip_header_card(key):
                continue
            if key in {"COMMENT", "HISTORY"}:
                continue
            merged_header[key] = (card.value, card.comment)

        return merged_header

    @classmethod
    def _should_skip_header_card(cls, key):
        return (
            key in cls._HEADER_MERGE_SKIP_KEYS
            or any(str(key).startswith(prefix) for prefix in cls._HEADER_MERGE_SKIP_PREFIXES)
        )

    def _ensure_filter_for_image(self, filepath, observatory, band):
        try:
            self.filter_inst.ensure_filter(
                name=band,
                facility=observatory,
                unknown_policy=self.unknown_policy,
                **self.filter_config,
            )
        except Exception as exc:
            if self.unknown_policy == "strict":
                raise
            if self.filter_config.get("warn", True):
                warnings.warn(
                    f"Filter curve unavailable for {observatory}.{band} from {filepath}: {exc}. "
                    "Keeping image in best-effort mode."
                )

    def update_data(self, image_data, galaxy_name, observatory, band):
        if not all([galaxy_name, observatory, band]):
            raise KeyError("Specify galaxy, observatory, and band.")
        try:
            self._data[galaxy_name][observatory][band] = image_data
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band specified")
        self._invalidate_last_catalog()

    def update_error(self, error_data, galaxy_name, observatory, band, source=None):
        if not all([galaxy_name, observatory, band]):
            raise KeyError("Specify galaxy, observatory, and band.")
        try:
            self._error[galaxy_name][observatory][band] = error_data
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band specified")
        if source is not None:
            self._error_sources().setdefault(galaxy_name, {}).setdefault(observatory, {})[
                band
            ] = str(source)
        self._invalidate_last_catalog()

    def get_error_source(self, galaxy_name, observatory, band):
        """Return the origin of an error map, defaulting to user-provided data."""
        try:
            error = self._error[galaxy_name][observatory][band]
        except KeyError:
            return "missing"
        if error is None:
            return "missing"

        source = (
            self._error_sources().get(galaxy_name, {})
            .get(observatory, {})
            .get(band)
        )
        if source in {None, "missing"}:
            source = "provided"
            self._error_sources().setdefault(galaxy_name, {}).setdefault(observatory, {})[
                band
            ] = source
        return source

    def has_missing_errors(self):
        """Return whether any loaded image still needs an error estimate."""
        for galaxy_name, observatories in self._data.items():
            for observatory, bands in observatories.items():
                for band in bands:
                    if self.get_error_source(galaxy_name, observatory, band) == "missing":
                        return True
        return False
        
    def update_header(self, updated_header, galaxy_name, observatory, band):
        if not all([galaxy_name, observatory, band]):
            raise KeyError("Specify galaxy, observatory, and band.")
        try:
            self._header[galaxy_name][observatory][band] = updated_header
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band specified")
        self._invalidate_last_catalog()

        
    # merging / append instance
    def append(self, other):
        """
        Append data from another GalaxyImageSet instance to this instance.
        This modifies the current instance.
        
        Args:
            other (GalaxyImageSet): Another GalaxyImageSet instance to append from
        """
        if not isinstance(other, GalaxyImageSet):
            raise TypeError("Can only append another GalaxyImageSet instance")

        collisions = {}
        for galaxy_name, observatories in other._data.items():
            for observatory, bands in observatories.items():
                for band in bands:
                    if band not in self._data.get(galaxy_name, {}).get(observatory, {}):
                        continue
                    key = (galaxy_name, observatory, band)
                    existing_path = (
                        self._files.get(galaxy_name, {})
                        .get(observatory, {})
                        .get(band, f"<in-memory:{'/'.join(key)}:existing>")
                    )
                    new_path = (
                        other._files.get(galaxy_name, {})
                        .get(observatory, {})
                        .get(band, f"<in-memory:{'/'.join(key)}:new>")
                    )
                    collisions[key] = (existing_path, new_path)
        if collisions:
            raise DuplicateImageError(collisions)
        
        # Merge data
        for galaxy_name, observatories in other._data.items():
            if galaxy_name not in self._data:
                self._data[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._data[galaxy_name]:
                    self._data[galaxy_name][observatory] = {}
                
                for band, image_data in bands.items():
                    self._data[galaxy_name][observatory][band] = image_data
        
        # Merge headers
        for galaxy_name, observatories in other._header.items():
            if galaxy_name not in self._header:
                self._header[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._header[galaxy_name]:
                    self._header[galaxy_name][observatory] = {}
                
                for band, header_data in bands.items():
                    self._header[galaxy_name][observatory][band] = header_data
        
        # Merge error data
        for galaxy_name, observatories in other._error.items():
            if galaxy_name not in self._error:
                self._error[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._error[galaxy_name]:
                    self._error[galaxy_name][observatory] = {}
                
                for band, error_data in bands.items():
                    self._error[galaxy_name][observatory][band] = error_data

        # Preserve whether error maps came from files or were estimated.
        error_sources = self._error_sources()
        for galaxy_name, observatories in getattr(other, "_error_source", {}).items():
            if galaxy_name not in error_sources:
                error_sources[galaxy_name] = {}
            for observatory, bands in observatories.items():
                if observatory not in error_sources[galaxy_name]:
                    error_sources[galaxy_name][observatory] = {}
                error_sources[galaxy_name][observatory].update(bands)
        
        # Merge PSF data
        for galaxy_name, observatories in other._psf.items():
            if galaxy_name not in self._psf:
                self._psf[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._psf[galaxy_name]:
                    self._psf[galaxy_name][observatory] = {}
                
                for band, psf_data in bands.items():
                    self._psf[galaxy_name][observatory][band] = psf_data
        
        # Merge cutout_shape data
        for galaxy_name, observatories in other._cutout_shape.items():
            if galaxy_name not in self._cutout_shape:
                self._cutout_shape[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._cutout_shape[galaxy_name]:
                    self._cutout_shape[galaxy_name][observatory] = {}
                
                for band, cutout_data in bands.items():
                    self._cutout_shape[galaxy_name][observatory][band] = cutout_data
        
        # Merge files data (nested dict structure)
        for galaxy_name, observatories in other._files.items():
            if galaxy_name not in self._files:
                self._files[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._files[galaxy_name]:
                    self._files[galaxy_name][observatory] = {}
                
                for band, filepath in bands.items():
                    self._files[galaxy_name][observatory][band] = filepath
        
        # Merge obs data
        for galaxy_name, obs_data in other._obs.items():
            if galaxy_name not in self._obs:
                self._obs[galaxy_name] = obs_data
            else:
                # If both have obs data for the same galaxy, merge them
                if isinstance(self._obs[galaxy_name], dict) and isinstance(obs_data, dict):
                    self._obs[galaxy_name].update(obs_data)
                else:
                    self._obs[galaxy_name] = obs_data
        self._invalidate_last_catalog()


    def merge(self, other):
        """
        Create a new GalaxyImageSet instance containing data from both instances.
        This does not modify either original instance.
        
        Args:
            other (GalaxyImageSet): Another GalaxyImageSet instance to merge with
            
        Returns:
            GalaxyImageSet: A new instance containing merged data
        """
        if not isinstance(other, GalaxyImageSet):
            raise TypeError("Can only merge with another GalaxyImageSet instance")
        
        # Create new instance
        merged = GalaxyImageSet(
            auto_ensure_filters=self.auto_ensure_filters,
            unknown_policy=self.unknown_policy,
            filter_config=self.filter_config,
        )
        
        # First append self to the new instance
        merged.append(self)
        
        # Then append other to the new instance
        merged.append(other)
        
        return merged
    

    @property
    def files(self):
        """Read Only: Path of Files"""
        return self._files
    
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value_tuple):
        galaxy, observatory, band, image_data = value_tuple
        try:
            self._data[galaxy][observatory][band] = image_data
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band for setting image data.")
        self._invalidate_last_catalog()
        
    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, value_tuple):
        galaxy, observatory, band, header_data = value_tuple
        try:
            self._header[galaxy][observatory][band] = header_data
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band for setting image data.")
        self._invalidate_last_catalog()
        
    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value_tuple):
        galaxy, observatory, band, image_data = value_tuple
        try:
            self._error[galaxy][observatory][band] = image_data
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band for setting image data.")
        self._error_sources().setdefault(galaxy, {}).setdefault(observatory, {})[
            band
        ] = "provided"
        self._invalidate_last_catalog()

    @property
    def error_source(self):
        """Read-only nested mapping describing the origin of each error map."""
        return self._error_sources()

    @property
    def psf(self):
        return self._psf
    
    @psf.setter
    def psf(self, value_tuple):
        if not isinstance(value_tuple, tuple):
            raise TypeError("psf.setter expects a tuple.")
        
        if len(value_tuple) == 4:
            galaxy, observatory, band, new_val = value_tuple
            if galaxy not in self._psf:
                self._psf[galaxy] = {}
            if observatory not in self._psf[galaxy]:
                self._psf[galaxy][observatory] = {}
            self._psf[galaxy][observatory][band] = new_val

        elif len(value_tuple) == 3:
            galaxy, val_name, new_val = value_tuple
            if galaxy not in self._psf:
                self._psf[galaxy] = {}
            self._psf[galaxy][val_name] = new_val

        else:
            raise ValueError("value_tuple must be length 3 or 4.")

    @property
    def cutout_shape(self):
        return self._cutout_shape

    @cutout_shape.setter
    def cutout_shape(self, value_tuple):
        if not hasattr(value_tuple, '__iter__'):
            raise TypeError("cutout_shape setter expects an iterable (tuple or list)")
        if len(value_tuple) != 4:
            raise ValueError("cutout_shape setter expects length-4 tuple: (galaxy, observatory, band, box_shape)")

        galaxy, observatory, band, box_shape = value_tuple

        if not (isinstance(box_shape, (list, tuple))):
            raise ValueError("box_shape must be tuple/list")

        if galaxy not in self._cutout_shape:
            self._cutout_shape[galaxy] = {}
        if observatory not in self._cutout_shape[galaxy]:
            self._cutout_shape[galaxy][observatory] = {}

        self._cutout_shape[galaxy][observatory][band] = box_shape

    def summary(self):
        """Print summary of galaxies and available bands using rich."""
        console = Console()

        for galaxy, observatories in self._data.items():
            gal_name = ", ".join(sorted(galaxy)) if isinstance(galaxy, list) else str(galaxy)

            tree = Tree(f"[bold deep_sky_blue2]Galaxy: {gal_name}[/bold deep_sky_blue2]")

            for obs, bands in observatories.items():
                obs_name = ", ".join(sorted(obs)) if isinstance(obs, list) else str(obs)
                band_list = ", ".join(sorted(bands.keys()))

                obs_node = tree.add(f"[bold dark_turquoise]Observatory: {obs_name}[/bold dark_turquoise]")
                obs_node.add(f"[cornflower_blue]Bands:[/cornflower_blue] {band_list}")

            console.print(tree)
            console.print()

    def ensure_filters(self, allow_svo=None, cache=None, cache_dir=None, warn=None, unknown_policy=None):
        """Preload filter curves required by the images in this set."""
        options = self.filter_config.copy()
        if allow_svo is not None:
            options["allow_svo"] = allow_svo
        if cache is not None:
            options["cache"] = cache
        if cache_dir is not None:
            options["cache_dir"] = cache_dir
        if warn is not None:
            options["warn"] = warn
        options["unknown_policy"] = unknown_policy or self.unknown_policy

        return self.filter_inst.ensure_filters_for_image_set(
            self,
            **options,
        )
                
    # ==== Plotting Properties ====
    def plot_image(self, value_tuple):
        if not hasattr(value_tuple, '__iter__'):
            raise TypeError("plot_image expects an iterable (tuple or list)")
        if len(value_tuple) != 3:
            raise ValueError("plot_image expects length-3 tuple: (galaxy, observatory, band)")
        
        galaxy, observatory, band = value_tuple
        try:
            fig, ax = DrawGalaxy.single_galaxy(self, galaxy, observatory, band)
            return fig, ax
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band for plotting image data.")

    def plot_sed(
        self,
        galaxy="*",
        *,
        row="total",
        galaxy_metadata=None,
        metadata_config=None,
        invalid_values=(99.0, -99.0),
    ):
        """Plot total or single-pixel SEDs from the latest pipeline catalog."""
        catalog_frame = self._require_last_catalog()
        return DrawGalaxy.plot_sed(
            self,
            catalog_frame,
            galaxy=galaxy,
            row=row,
            galaxy_metadata=galaxy_metadata,
            metadata_config=metadata_config,
            invalid_values=invalid_values,
        )

    def write_catalog(
        self,
        output_path,
        *,
        format=None,
        include_total=False,
        galaxy_metadata=None,
        metadata_config=None,
        invalid_values=(99.0, -99.0),
        overwrite=False,
    ):
        """Write the latest pipeline catalog as per-galaxy FITS/ASCII tables."""
        catalog_frame = self._require_last_catalog()
        return write_image_set_catalog(
            self,
            catalog_frame,
            output_path,
            format=format,
            include_total=include_total,
            galaxy_metadata=galaxy_metadata,
            metadata_config=metadata_config,
            invalid_values=invalid_values,
            overwrite=overwrite,
        )
        

class Parsers:
    _KNOWN_BANDS_BY_OBSERVATORY = {
        "GALEX": {"FUV", "NUV"},
        "SDSS": {"u", "g", "r", "i", "z"},
        "PACS": {"blue", "green", "red"},
        "SPIRE": {"PSW", "PMW", "PLW"},
        "WISE": {"W1", "W2", "W3", "W4", "w1", "w2", "w3", "w4"},
        "UKIRT": {"Y", "J", "H", "K"},
    }
    _HEADER_OBSERVATORY_KEYS = ("INSTRUME", "INSTRUMENT", "TELESCOP", "OBSERVAT", "FACILITY", "OBS")
    _HEADER_BAND_KEYS = ("FILTER", "FILTER1", "FILTER2", "FILTERID", "BAND", "BANDPASS")

    def __init__(self):
        pass

    @staticmethod
    def _split_tokens(value):
        if value is None:
            return []
        pattern = "|".join(map(re.escape, ['-', ' ', '_', '.', '/', ':']))
        return [token for token in re.split(pattern, str(value).strip()) if token]

    @staticmethod
    def _header_values(header, keys):
        if header is None:
            return []

        values = []
        for key in keys:
            try:
                value = header.get(key)
            except AttributeError:
                value = header[key] if key in header else None
            if value is None:
                continue
            value = str(value).strip()
            if value and value.upper() not in {"NONE", "UNKNOWN", "N/A", "NULL"}:
                values.append(value)
        return values

    @classmethod
    def _known_observatory_from_value(cls, value):
        candidates = [str(value).strip()] + cls._split_tokens(value)
        for candidate in candidates:
            candidate = Observatories.normalize_name(candidate)
            for obs in Observatories.get_observatories():
                if candidate.lower() == obs.lower():
                    return obs
        return None

    @classmethod
    def _known_band_from_value(cls, value, filter_inst, observatory=None):
        candidates = [str(value).strip()] + cls._split_tokens(value)
        known_bands = list(filter_inst.get_all_filters())
        if observatory in cls._KNOWN_BANDS_BY_OBSERVATORY:
            known_bands.extend(cls._KNOWN_BANDS_BY_OBSERVATORY[observatory])

        for candidate in candidates:
            for band in known_bands:
                if candidate.lower() == str(band).lower():
                    return band
        return None

    @classmethod
    def _observatory_name_parser(cls, file_name, header=None, best_effort=False):
        """Parse observatory names from a given string or list."""
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")
        parts = cls._split_tokens(file_name)
        for part in parts:
            known = cls._known_observatory_from_value(part)
            if known is not None:
                return known

        header_values = cls._header_values(header, cls._HEADER_OBSERVATORY_KEYS)
        for value in header_values:
            known = cls._known_observatory_from_value(value)
            if known is not None:
                return known
        if best_effort and header_values:
            return header_values[0]
        if best_effort and len(parts) >= 3:
            return parts[-2]
        return None
    
    @staticmethod
    def _galaxy_name_parser(file_name):
        """Parse galaxy names from a given string or list."""
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")
        for galaxy_category in ['NGC', 'IC', 'M']:
            pattern = "|".join(map(re.escape, ['-', ' ', '_', '.']))
            match = [g for g in re.split(pattern, file_name) if galaxy_category in g]
            if match:
                return match[0]
        return None
    
    @classmethod
    def _band_name_parser(cls, file_name, filter_inst, header=None, observatory=None, best_effort=False):
        """Parse band names from a given string or list."""
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")

        parts = cls._split_tokens(file_name)
        
        for band in filter_inst.get_all_filters():
            if any(str(band).lower() == part.lower() for part in parts):
                return band

        observatory = observatory or cls._observatory_name_parser(file_name)
        for band in cls._KNOWN_BANDS_BY_OBSERVATORY.get(observatory, set()):
            if any(str(band).lower() == part.lower() for part in parts):
                return band

        header_values = cls._header_values(header, cls._HEADER_BAND_KEYS)
        for value in header_values:
            known = cls._known_band_from_value(value, filter_inst, observatory=observatory)
            if known is not None:
                return known
        if best_effort and header_values:
            return header_values[0]
        if best_effort and len(parts) >= 2:
            return parts[-1]
        return None
    

class ImageQuery():
    def __init__(self):
        pass    
                
    def queryImages(self, dir, galaxy_name=None, band=None):
        """
        Query images from a directory based on galaxy names and bands.
        Returns a GalaxyImageSet object containing the queried images.
        """
        image_set = GalaxyImageSet()
        dir_path = Path(dir).parent
        file_pattern = Path(dir).name if Path(dir).name.endswith('.fits') else "*.fits"
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"{dir} is not a valid directory")

        if isinstance(galaxy_name, str):
            galaxy_name = [galaxy_name]
        if isinstance(band, str):
            band = [band]

        collisions = {}
        for fits_file in sorted(dir_path.glob(file_pattern)):
            try:
                image_set.add_image(fits_file)
            except DuplicateImageError as exc:
                for key, paths in exc.collisions.items():
                    collisions.setdefault(key, set()).update(paths)
            except Exception as e:
                emit_alert(
                    f"Could not process FITS input: {e}",
                    context=str(fits_file),
                    dedupe_key=f"image_query.{type(e).__name__}.{e}",
                )
        if collisions:
            raise DuplicateImageError(collisions)
                    
        return image_set

    @classmethod
    def queryAllImages(cls, dir):
        image_set = GalaxyImageSet()
        
        if not isinstance(dir, str):
            dir = str(dir)
        
        collisions = {}
        for file in sorted(glob(dir)):
            if Path(file).stem.endswith("_err"):
                continue
            try:
                image_set.add_image(file)
            except DuplicateImageError as exc:
                for key, paths in exc.collisions.items():
                    collisions.setdefault(key, set()).update(paths)
        if collisions:
            raise DuplicateImageError(collisions)
            
        return image_set
