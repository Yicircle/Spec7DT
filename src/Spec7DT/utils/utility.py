import numpy as np
import math
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog

class Filters:
    """Class to handle different photometric filters and their properties."""
    def __init__(self):
        self.broadband = self._broadband()
        self.mediumband = self._mediumband()
        self.filters = self.broadband + self.mediumband
    
    def _broadband(self):
        """Return a list of broadband filters."""
        return [
            'FUV', 'NUV', 'u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'Ks',
            'w1', 'w2', 'w3', 'w4'
        ]
    
    def _mediumband(self):
        """Return a list of mediumband filters."""
        return [f'm{wave}' for wave in range(400, 900, 25)]
    
    @classmethod
    def get_filters(cls):
        """Return a list of all filters."""
        return cls().filters
    
    @classmethod
    def get_catcols(cls, cat_type):
        """Return a dictionary of given type"""
        catcols ={"cigale": cls.cigale(cls),
                  "eazy": cls.eazy(cls),
                  "lephare": cls.lephare(cls),
                  "ppxf": cls.ppxf(cls),
                  "goyangyi": cls.goyangyi(cls)
                  }
        
        return catcols[cat_type.lower()]
    
    
    def cigale(self):
        cols_cigale = {
            'GALEX.NUV': 'galex.NUV',
            'GALEX.FUV': 'galex.FUV',
            'SDSS.u': 'SDSS_u',
            'SDSS.g': 'SDSS_g',
            'SDSS.r': 'SDSS_r',
            'SDSS.i': 'SDSS_i',
            'SDSS.z': 'SDSS_z',
            'PanStarrs.y': 'PAN-STARRS_y',
            '2MASS.J': 'J_2mass',
            '2MASS.H': 'H_2mass',
            '2MASS.Ks': 'Ks_2mass',
            'Spitzer.ch1': 'spitzer.irac.ch1',
            'Spitzer.ch2': 'spitzer.irac.ch2',
            'WISE.w1': 'WISE1',
            'WISE.w2': 'WISE2',
            'WISE.w3': 'WISE3',
            'WISE.w4': 'WISE4',
            'F657N': 'HST.UVIS1.F657N',
            'F658N': 'HST.UVIS1.F658N',
        }
        cols_cigale.update({f'{key}_err': f'{cols_cigale[key]}_err' for key in cols_cigale.keys() if '_err' not in key})
        return cols_cigale
    
    def eazy(self):
        cols_eazy = {
            
        }
        return cols_eazy
    
    def lephare(self):
        cols_lephare = {
            
        }
        return cols_lephare
    
    def ppxf(self):
        cols_ppxf = {
            
        }
        return cols_ppxf
    
    
    def goyangyi(self):
        print(" ╱|、\n(˚ˎ 。7  \n |、˜〵          \n じしˍ,)ノ")
        return self.cigale(self)

class Observatories:
    """Class to handle different observatories and their properties."""
    def __init__(self):
        self.optical_obs = self._opticals()
        self.ir_obs = self._infrareds()
        self.uv_obs = self._ultraviolet()
        self.radio_obs = self._radio()
        self.observatories = list(set(self.optical_obs + self.ir_obs + self.uv_obs + self.radio_obs))
        
    def _opticals(self):
        """Return a list of optical observatories."""
        return ['HST', 'SDSS', 'PS1', 'CFHT', 'DECam', 'DES', 'LSST', 'Pan-STARRS', 'Subaru', '7DT', 'SkyMapper']
    
    def _infrareds(self):
        """Return a list of infrared observatories."""
        return ['WISE', 'Spitzer', 'Herschel', 'JWST', 'VISTA', 'UKIDSS', '2MASS']
    
    def _ultraviolet(self):
        """Return a list of ultraviolet observatories."""
        return ['GALEX', 'HST', 'FUSE']
    
    def _radio(self):
        """Return a list of radio observatories."""
        return ['VLA', 'ALMA', 'LOFAR', 'SKA', 'MeerKAT', 'GMRT']
    
    @classmethod
    def get_observatories(cls):
        """Return a list of all observatories."""
        return cls().observatories

class useful_functions:
    
    @classmethod
    def get_galaxy_radius(cls, image):
        
        threshold = detect_threshold(image, nsigma=3.0)

        segm = detect_sources(image, threshold, npixels=5)

        catalog = SourceCatalog(image, segm)
        gal = max(catalog, key=lambda src: src.area)

        x0, y0 = gal.xcentroid, gal.ycentroid
        a, b = gal.semimajor_sigma.value*2, gal.semiminor_sigma.value*2
        theta = math.radians(gal.orientation.value)
        return x0, y0, a, b, theta
    
    @staticmethod
    def extract_values_recursive(dictionary, key):
        """
        Alternative recursive approach that handles arbitrary nesting depth.
        
        Args:
            dictionary: Dictionary with nested structure
            key: The key at level1 to extract values from
        
        Returns:
            List of all values found in the nested structure
        """
        def _extract_all_values(obj):
            """Recursively extract all values from nested dict/list structures."""
            if isinstance(obj, dict):
                values = []
                for v in obj.values():
                    values.extend(_extract_all_values(v))
                return values
            elif isinstance(obj, list):
                values = []
                for item in obj:
                    values.extend(_extract_all_values(item))
                return values
            else:
                return [obj]
        
        if key not in dictionary:
            return []
        
        return _extract_all_values(dictionary[key])
    
    @classmethod
    def tour_nested_dict_with_keys(cls, dictionary):
        """
        Tour through a 3-level nested dictionary and yield keys and values in order.
        
        Args:
            dictionary: Dictionary with structure dict[level1][level2][level3]
        
        Yields:
            tuple: (keys_tuple, value) where keys_tuple contains (level1_key, level2_key, level3_key)
        """
        for level1_key, level1_dict in dictionary.items():
            for level2_key, level2_dict in level1_dict.items():
                for level3_key, value in level2_dict.items():
                    yield (level1_key, level2_key, level3_key), value


    def get_all_keys_and_values(self, my_dict):
        """
        Get all keys and values from a 3-level nested dictionary as a list.
        
        Args:
            my_dict: Dictionary with structure dict[level1][level2][level3]
        
        Returns:
            list: List of tuples [(keys_tuple, value), ...] where keys_tuple contains (level1_key, level2_key, level3_key)
        """
        result = []
        for level1_key, level1_dict in my_dict.items():
            for level2_key, level2_dict in level1_dict.items():
                for level3_key, value in level2_dict.items():
                    result.append(((level1_key, level2_key, level3_key), value))
        return result


    def tour_nested_dict_recursive(self, obj, current_keys=()):
        """
        Recursive function to tour through arbitrarily nested dictionaries.
        
        Args:
            obj: Dictionary or value to traverse
            current_keys: Current key path (used internally)
        
        Yields:
            tuple: (keys_tuple, value) where keys_tuple contains all keys in the path
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield from self.tour_nested_dict_recursive(value, current_keys + (key,))
        else:
            yield current_keys, obj