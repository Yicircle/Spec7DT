import numpy as np
from pathlib import Path
from astropy.io import fits
import pandas as pd
import re
import inspect
import warnings
from glob import glob
from functools import wraps
from typing import Callable, Optional, Union, List

from .utility import Observatories
from .utility import useful_functions
from .file_generator import *
from ..plot.plot import DrawGalaxy
from ..handlers.filter_handler import Filters


class GalaxyImageSet:
    __signature__ = inspect.Signature()

    def __init__(self):
        self._data = {}
        self._header = {}
        self._error = {}
        self._psf = {}
        self._cutout_shape = {}
        self._obs = {}
        self._files = []
        self.filter_inst = Filters()
        
        
    def add_image(self, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        with fits.open(filepath) as hdul:
            image_data = hdul[0].data
            image_header = hdul[0].header

        
        file_name = filepath.stem
        galaxy_name = Parsers._galaxy_name_parser(file_name=file_name)
        band = Parsers._band_name_parser(file_name=file_name, filter_inst=self.filter_inst)
        observatory = Parsers._observatory_name_parser(file_name=file_name)
        if galaxy_name is None or band is None or observatory is None:
            return

        for attr in [self._data, self._header, self._error, self._psf, self._cutout_shape]:
            attr.setdefault(galaxy_name, {}).setdefault(observatory, {})

        self._data[galaxy_name][observatory][band] = image_data
        self._header[galaxy_name][observatory][band] = image_header
        self._psf[galaxy_name][observatory][band] = -1.0
        self._cutout_shape[galaxy_name][observatory][band] = set()
        self._files.append(str(filepath))

        err_name = filepath.with_name(filepath.stem + "_err").with_suffix(filepath.suffix)
        if not err_name.exists():
            warnings.warn(f"Error File {err_name} not found. Continue without error map.")
            self._error[galaxy_name][observatory][band] = np.zeros_like(image_data)
        else:
            with fits.open(err_name) as hdul_err:
                error_data = hdul_err[0].data
            self._error[galaxy_name][observatory][band] = error_data

    def update_data(self, image_data, galaxy_name, observatory, band):
        if not all([galaxy_name, observatory, band]):
            raise KeyError("Specify galaxy, observatory, and band.")
        try:
            self._data[galaxy_name][observatory][band] = image_data
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band specified")

    def update_error(self, error_data, galaxy_name, observatory, band):
        if not all([galaxy_name, observatory, band]):
            raise KeyError("Specify galaxy, observatory, and band.")
        try:
            self._error[galaxy_name][observatory][band] = error_data
        except KeyError:
            raise KeyError("Invalid galaxy/observatory/band specified")
    
        
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
        
        # Merge data
        for galaxy_name, observatories in other._data.items():
            if galaxy_name not in self._data:
                self._data[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._data[galaxy_name]:
                    self._data[galaxy_name][observatory] = {}
                
                for band, image_data in bands.items():
                    # Override if exists, add if new
                    self._data[galaxy_name][observatory][band] = image_data
        
        # Merge headers
        for galaxy_name, observatories in other._header.items():
            if galaxy_name not in self.header:
                self._header[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._header[galaxy_name]:
                    self._header[galaxy_name][observatory] = {}
                
                for band, header_data in bands.items():
                    self._header[galaxy_name][observatory][band] = header_data
        
        # Merge error data
        for galaxy_name, observatories in other.error.items():
            if galaxy_name not in self.error:
                self._error[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._error[galaxy_name]:
                    self._error[galaxy_name][observatory] = {}
                
                for band, error_data in bands.items():
                    self._error[galaxy_name][observatory][band] = error_data
        
        # Merge PSF data
        for galaxy_name, observatories in other._psf.items():
            if galaxy_name not in self._psf:
                self._psf[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self._psf[galaxy_name]:
                    self._psf[galaxy_name][observatory] = {}
                
                for band, psf_data in bands.items():
                    self._psf[galaxy_name][observatory][band] = psf_data
        
        # Merge obs data
        for galaxy_name, obs_data in other._obs.items():
            if galaxy_name not in self.obs:
                self._obs[galaxy_name] = obs_data
            else:
                # If both have obs data for the same galaxy, merge them
                if isinstance(self._obs[galaxy_name], dict) and isinstance(obs_data, dict):
                    self._obs[galaxy_name].update(obs_data)
                else:
                    self._obs[galaxy_name] = obs_data
        
        # Merge file lists (avoid duplicates)
        for filepath in other._files:
            if filepath not in self._files:
                self._files.append(filepath)


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
        merged = GalaxyImageSet()
        
        # First append self to the new instance
        merged.append(self)
        
        # Then append other to the new instance
        merged.append(other)
        
        return merged
    

    @property
    def files(self):
        """Read Only: Path of Files"""
        return tuple(self._files)
    
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
        
        print(f"Set cutout_shape[{galaxy}][{observatory}][{band}] = {box_shape}")
        
    def summary(self):
        """Print summary of galaxies and available bands."""
        for galaxy, val in self._data.items():
            for obs, bands in val.items():
                galaxy = sorted(galaxy) if isinstance(galaxy, list) else [galaxy]
                obs = sorted(obs) if isinstance(obs, list) else [obs]
                print(f"Galaxy: {galaxy}\nObservatories: {obs} \nBands: {sorted(bands.keys())}")
                
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
        

class Parsers:
    def __init__(self):
        pass
    
    @staticmethod
    def _observatory_name_parser(file_name):
        """Parse observatory names from a given string or list."""
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")
        for obs in Observatories.get_observatories():
            pattern = "|".join(map(re.escape, ['-', ' ', '_', '.']))
            if obs in re.split(pattern, file_name):
                return obs
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
    
    @staticmethod
    def _band_name_parser(file_name, filter_inst):
        """Parse band names from a given string or list."""
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")
        
        for band in filter_inst.get_all_filters():
            pattern = "|".join(map(re.escape, ['-', ' ', '_', '.']))
            if band in re.split(pattern, file_name):
                return band
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

        for fits_file in dir_path.glob(file_pattern):
            try:
                image_set.add_image(fits_file)
            except Exception as e:
                print(f"Error processing {fits_file}: {e}")
                    
        return image_set
    
    @classmethod
    def queryAllImages(cls, dir):
        image_set = GalaxyImageSet()
        
        if not isinstance(dir, str):
            dir = str(dir)
        
        for file in list(glob(dir)):
            if Path(file).stem.endswith("_err"):
                continue
            image_set.add_image(file)
            
        return image_set