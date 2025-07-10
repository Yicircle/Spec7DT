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

from .utility import Filters, Observatories
from .utility import useful_functions
from utils.file_generator import *


class GalaxyImageSet():
    __signature__ = inspect.Signature()
    
    def __init__(self):
        self.data = {}
        self.header = {}
        self.error = {}
        self.psf = {}
        self.obs = {}
        self.files = []
        
    def add_image(self, filepath):
        """
        Add an image to the dataset under a galaxy and band identifier.
        Automatically loads the FITS data into memory.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        with fits.open(filepath) as hdul:
            image_data = hdul[0].data
            image_header = hdul[0].header

        # Parse galaxy name, band, and observatory from the filename
        file_name = filepath.stem
        galaxy_name = Parsers._galaxy_name_parser(file_name=file_name)
        band = Parsers._band_name_parser(file_name=file_name)
        observatory = Parsers._observatory_name_parser(file_name=file_name)
        
        if galaxy_name is None or band is None or observatory is None:
            return
        
        if galaxy_name not in self.data:
            self.data[galaxy_name] = {}
            self.header[galaxy_name] = {}
            self.error[galaxy_name] = {}
            self.psf[galaxy_name] = {}
            
        if observatory not in self.data[galaxy_name]:
            self.data[galaxy_name][observatory] = {}
            self.header[galaxy_name][observatory] = {}
            self.error[galaxy_name][observatory] = {}
            self.psf[galaxy_name][observatory] = {}
                
        self.data[galaxy_name][observatory][band] = image_data
        self.header[galaxy_name][observatory][band] = image_header
        self.psf[galaxy_name][observatory][band] = -1.0
        self.files.append(str(filepath))
        
        # Find error map and add to variable
        err_name = filepath.with_name(filepath.stem + "_err").with_suffix(filepath.suffix)
        
        if not err_name.exists():
            warnings.warn(f"Error File {err_name} not found. Continue without error map.")
            self.error[galaxy_name][observatory][band] = np.zeros_like(image_data)
        else:
            with fits.open(err_name) as hdul_err:
                error_data = hdul_err[0].data
            
            self.error[galaxy_name][observatory][band] = error_data
        
    # Update image data
    def update_data(self, image_data, galaxy_name, observatory, band):
        """
        Update an image to the dataset under a galaxy and band identifier.
        """
        # print(self.data[galaxy_name][observatory][band].shape)
        # print(np.nansum(self.data[galaxy_name][observatory][band]))
        
        if galaxy_name is None or band is None or observatory is None:
            raise KeyError("Specify the galaxy name, band and observatory name.")
        
        if (galaxy_name not in self.data 
            or observatory not in self.data[galaxy_name] 
            or band not in self.data[galaxy_name][observatory]):
            raise KeyError("Input valid name.")
                
        self.data[galaxy_name][observatory][band] = image_data
        # print(self.data[galaxy_name][observatory][band].shape)
        # print(np.nansum(self.data[galaxy_name][observatory][band]))
        
    def update_error(self, error_data, galaxy_name, observatory, band):
        """
        Update an image to the dataset under a galaxy and band identifier.
        """
        
        if galaxy_name is None or band is None or observatory is None:
            raise KeyError("Specify the galaxy name, band and observatory name.")
        
        if (galaxy_name not in self.error 
            or observatory not in self.error[galaxy_name] 
            or band not in self.error[galaxy_name][observatory]):
            raise KeyError("Input valid name.")
                
        self.error[galaxy_name][observatory][band] = error_data
        
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
        for galaxy_name, observatories in other.data.items():
            if galaxy_name not in self.data:
                self.data[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self.data[galaxy_name]:
                    self.data[galaxy_name][observatory] = {}
                
                for band, image_data in bands.items():
                    # Override if exists, add if new
                    self.data[galaxy_name][observatory][band] = image_data
        
        # Merge headers
        for galaxy_name, observatories in other.header.items():
            if galaxy_name not in self.header:
                self.header[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self.header[galaxy_name]:
                    self.header[galaxy_name][observatory] = {}
                
                for band, header_data in bands.items():
                    self.header[galaxy_name][observatory][band] = header_data
        
        # Merge error data
        for galaxy_name, observatories in other.error.items():
            if galaxy_name not in self.error:
                self.error[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self.error[galaxy_name]:
                    self.error[galaxy_name][observatory] = {}
                
                for band, error_data in bands.items():
                    self.error[galaxy_name][observatory][band] = error_data
        
        # Merge PSF data
        for galaxy_name, observatories in other.psf.items():
            if galaxy_name not in self.psf:
                self.psf[galaxy_name] = {}
            
            for observatory, bands in observatories.items():
                if observatory not in self.psf[galaxy_name]:
                    self.psf[galaxy_name][observatory] = {}
                
                for band, psf_data in bands.items():
                    self.psf[galaxy_name][observatory][band] = psf_data
        
        # Merge obs data
        for galaxy_name, obs_data in other.obs.items():
            if galaxy_name not in self.obs:
                self.obs[galaxy_name] = obs_data
            else:
                # If both have obs data for the same galaxy, merge them
                if isinstance(self.obs[galaxy_name], dict) and isinstance(obs_data, dict):
                    self.obs[galaxy_name].update(obs_data)
                else:
                    self.obs[galaxy_name] = obs_data
        
        # Merge file lists (avoid duplicates)
        for filepath in other.files:
            if filepath not in self.files:
                self.files.append(filepath)


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


    # Get data
    def get_galaxy_data(self, galaxy_name):
        """Return all band images for a specific galaxy."""
        return self.data.get(galaxy_name, {})
    
    def get_observatory_image(self, galaxy_name, observatory):
        """Return image of a specific observatory for a specific galaxy"""
        return self.data.get(galaxy_name, {}).get(observatory, None)
    
    def get_band_image(self, galaxy_name, observatory, band):
        """Return image of a specific band for a specific galaxy."""
        return self.data.get(galaxy_name, {}).get(observatory, {}).get(band, None)
    
    def get_header(self, galaxy_name=None, band=None):
        """Return header information for a specific galaxy or band."""
        return self.header.get(galaxy_name, {}).get(band, None) if galaxy_name else self.header

    def get_psf(self, galaxy_name=None):
        return useful_functions.extract_values_recursive(self.psf, galaxy_name)
    
    
    # Set data
    def set_psf(self, galaxy_name, observatory, band, new_psf_value):
        self.psf[galaxy_name][observatory][band] = new_psf_value

    def summary(self) -> None:
        """Print summary of galaxies and available bands."""
        for galaxy, val in self.data.items():
            for obs, bands in val.items():
                galaxy = sorted(galaxy) if isinstance(galaxy, list) else [galaxy]
                obs = sorted(obs) if isinstance(obs, list) else [obs]
                print(f"Galaxy: {galaxy}\nObservatories: {obs} \nBands: {sorted(bands.keys())}")
                
    # Get file from instance
    def get_input_file(self):
        pass
        

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
    def _band_name_parser(file_name):
        """Parse band names from a given string or list."""
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")
        for band in Filters.get_filters():
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