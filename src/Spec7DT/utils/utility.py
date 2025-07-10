import numpy as np

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