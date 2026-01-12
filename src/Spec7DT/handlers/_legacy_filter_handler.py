from typing import Union, Tuple, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from importlib import resources

import numpy as np
from astroquery.svo_fps import SvoFps


@dataclass
class FilterCurve:
    """Container for filter response curve data."""
    name: str
    wavelength: np.ndarray
    response: np.ndarray
    source_type: str  # 'default', 'file', 'array'
    source_path: Optional[str] = None
    unit_type: str = 'photon'  # 'photon' or 'energy'
    description: str = ''
    
    def __post_init__(self):
        """Validate and convert arrays to numpy."""
        self.wavelength = np.asarray(self.wavelength)
        self.response = np.asarray(self.response)
        
        if len(self.wavelength) != len(self.response):
            raise ValueError("Wavelength and response arrays must have same length")
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save filter curve to ASCII .dat file with proper header format."""
        filepath = Path(filepath)
        
        # Create header lines
        header_lines = [
            f"# {self.name}",
            f"# {self.unit_type}",
            f"# {self.description}"
        ]
        
        # Write file manually to control header format exactly
        with open(filepath, 'w') as f:
            # Write header
            for line in header_lines:
                f.write(line + '\n')
            
            # Write data
            for wl, resp in zip(self.wavelength, self.response):
                f.write(f"{wl:.3f} {resp:.3f}\n")

class Filters:
    """Class to handle different photometric filters and their properties."""
    
    # Default filter names (curves loaded lazily)
    _default_broadband = [
        'FUV', 'NUV', 'u', 'g', 'r', 'i', 'z', 'Y', 'J', 'H', 'Ks',
        'w1', 'w2', 'w3', 'w4', "ch1", "ch2", "ch3", "ch4", "24mu", "red", "green", "blue", "PSW", "PMW", "PLW"
    ]
    _default_mediumband = [f'm{wave}' for wave in range(400, 900, 25)]
    
    # Storage for loaded filter curves
    _loaded_curves: Dict[str, FilterCurve] = {}
    _custom_filters: Dict[str, FilterCurve] = {}
    
    def __init__(self):
        """Initialize filter instance."""
        self.broadband = self._default_broadband.copy()
        self.mediumband = self._default_mediumband.copy()
        self.filters = self.broadband + self.mediumband + list(self._custom_filters.keys())
        self._facilities = self._query_facilities_and_instruments()
    
    @classmethod
    def _load_default_filter(cls, filter_name: str) -> FilterCurve:
        """Load a default filter from package data."""
        if filter_name in cls._loaded_curves:
            return cls._loaded_curves[filter_name]
        
        try:
            # Get the filter curves directory
            filter_dir = resources.files("Spec7DT.reference.filter_curves")
            
            # Search for files that contain the filter name
            matching_files = []
            with resources.as_file(filter_dir) as dir_path:
                if dir_path.is_dir():
                    for file_path in dir_path.glob("*.dat"):
                        # Check if filter_name is in the filename
                        if f".{filter_name}." in file_path.name or file_path.stem.endswith(f".{filter_name}"):
                            matching_files.append(file_path)
                    
                    # If no exact match, try looser matching
                    if not matching_files:
                        for file_path in dir_path.glob("*.dat"):
                            if filter_name in file_path.name:
                                matching_files.append(file_path)
            
            if not matching_files:
                raise FileNotFoundError(f"No filter file found containing '{filter_name}'")
            
            if len(matching_files) > 1:
                file_names = [f.name for f in matching_files]
                print(f"Warning: Multiple files found for '{filter_name}': {file_names}")
                print(f"Using: {matching_files[0].name}")
            
            # Use the first matching file
            selected_file = matching_files[0]
            
            # Load data and header information
            wavelength, response, file_filter_name, unit_type, description = cls._load_file_filter(selected_file)
                
            curve = FilterCurve(
                name=filter_name,  # Use requested name, not file header name
                wavelength=wavelength,
                response=response,
                source_type='default',
                source_path=str(selected_file),
                unit_type=unit_type,
                description=description
            )
            
            cls._loaded_curves[filter_name] = curve
            return curve
            
        except Exception as e:
            raise FileNotFoundError(f"Could not load default filter '{filter_name}': {e}")
    
    @classmethod
    def _load_file_filter(cls, filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, str, str, str]:
        """Load filter data from ASCII .dat file and parse header."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Filter file not found: {filepath}")
        
        try:
            # Read header information
            filter_name = ""
            unit_type = "photon"  # default
            description = ""
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            # Parse first three header lines
            header_count = 0
            
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    content = line[1:].strip()  # Remove # and whitespace
                    if header_count == 0:
                        filter_name = content
                    elif header_count == 1:
                        unit_type = content if content in ['photon', 'energy'] else 'photon'
                    elif header_count == 2:
                        description = content
                    header_count += 1
                else:
                    break
            
            # Use filename stem if no filter name in header
            if not filter_name:
                filter_name = filepath.stem
            
            # Load numerical data
            data = np.loadtxt(filepath)
            if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError("File must contain 2 columns: wavelength and response")
                
            return data[:, 0], data[:, 1], filter_name, unit_type, description
            
        except Exception as e:
            raise ValueError(f"Could not read filter file '{filepath}': {e}")
    
    @classmethod
    def add_filter_from_file(cls, filepath: Union[str, Path], filter_name: Optional[str] = None, verbose=False):
        """Add custom filter from ASCII .dat file."""
        filepath = Path(filepath)
        
        wavelength, response, file_filter_name, unit_type, description = cls._load_file_filter(filepath)
        
        # Use provided name, or file header name, or filename as fallback
        final_name = filter_name or file_filter_name or filepath.stem
        
        curve = FilterCurve(
            name=final_name,
            wavelength=wavelength,
            response=response,
            source_type='file',
            source_path=str(filepath.resolve()),
            unit_type=unit_type,
            description=description
        )
        
        cls._custom_filters[final_name] = curve
        if verbose:
            print(f"Added filter '{final_name}' from file: {filepath}")
            if description:
                print(f"  Description: {description}")
                print(f"  Unit type: {unit_type}")
    
    @classmethod
    def add_filter_from_arrays(cls, wavelength: Union[List, np.ndarray], 
                              response: Union[List, np.ndarray], filter_name: str,
                              unit_type: str = 'photon', description: str = ''):
        """Add custom filter from numpy arrays or lists."""
        curve = FilterCurve(
            name=filter_name,
            wavelength=wavelength,
            response=response,
            source_type='array',
            unit_type=unit_type,
            description=description
        )
        
        cls._custom_filters[filter_name] = curve
        print(f"Added filter '{filter_name}' from arrays")
    
    @classmethod
    def remove_filter(cls, filter_name: str) -> bool:
        """Remove a custom filter."""
        if filter_name in cls._custom_filters:
            del cls._custom_filters[filter_name]
            print(f"Removed custom filter: {filter_name}")
            return True
        else:
            print(f"Filter '{filter_name}' not found in custom filters")
            return False
    
    @classmethod
    def get_filter_curve(cls, filter_name: str, observatory: Optional[str] = None) -> FilterCurve:
        """Get filter response curve."""        
        try:
            for key, values in cls._facilities.items():
                if observatory in values:
                    facility = key
                    instrument = observatory
                    break
            else:
                if observatory in cls._facilities.keys():
                    facility = observatory
                    instrument = observatory

            data = SvoFps.get_transmission_data(f'{facility}/{instrument}.{filter_name}')
            
            curve = FilterCurve(
                    name=f"{observatory}.{filter_name}",
                    wavelength=data['Wavelength'].array,
                    response=data['Transmission'].array,
                    source_type='default',
                    source_path="SVO Filter Service",
                    unit_type=unit_type,
                    description=description
                )
            
            return curve
            
        except:
            # Check custom filters first
            if filter_name in cls._custom_filters:
                return cls._custom_filters[filter_name]
            
            elif filter_name in cls._default_broadband + cls._default_mediumband:
                if observatory:
                    return cls._load_default_filter_with_observatory(filter_name, observatory)
                else:
                    return cls._load_default_filter(filter_name)
        
        raise ValueError(f"Filter '{filter_name}' not found")
    
    @classmethod
    def _load_default_filter_with_observatory(cls, filter_name: str, observatory: str) -> FilterCurve:
        """Load a specific observatory filter."""
        cache_key = f"{observatory}.{filter_name}"
        
        if cache_key in cls._loaded_curves:
            return cls._loaded_curves[cache_key]
        
        try:
            # Look specifically for observatory.filter.dat
            filter_dir = resources.files("Spec7DT.reference.filter_curves")
            target_filename = f"{observatory}.{filter_name}.dat"
            
            with resources.as_file(filter_dir / target_filename) as dat_file:
                if dat_file.exists():
                    wavelength, response, file_filter_name, unit_type, description = cls._load_file_filter(dat_file)
                    
                    curve = FilterCurve(
                        name=f"{observatory}.{filter_name}",
                        wavelength=wavelength,
                        response=response,
                        source_type='default',
                        source_path=str(dat_file),
                        unit_type=unit_type,
                        description=description
                    )
                    
                    cls._loaded_curves[cache_key] = curve
                    return curve
                else:
                    raise FileNotFoundError(f"Specific filter file not found: {target_filename}")
            
        except Exception as e:
            # Fall back to general filter search
            print(f"Could not find {observatory}.{filter_name}.dat, trying general search...")
            return cls._load_default_filter(filter_name)
    
    @classmethod
    def get_all_filters(cls) -> List[str]:
        """Return list of all available filters."""
        return cls._default_broadband + cls._default_mediumband + list(cls._custom_filters.keys())
    
    @classmethod
    def get_custom_filters(cls) -> List[str]:
        """Return list of custom filters only."""
        return list(cls._custom_filters.keys())
    
    @classmethod
    def interpolate_filter(cls, filter_name: str, new_wavelength: np.ndarray) -> np.ndarray:
        """Interpolate filter response to new wavelength grid."""
        curve = cls.get_filter_curve(filter_name)
        return np.interp(new_wavelength, curve.wavelength, curve.response, left=0, right=0)
    
    @classmethod
    def save_custom_filter(cls, filter_name: str, filepath: Union[str, Path]):
        """Save a custom filter to file."""
        if filter_name not in cls._custom_filters:
            raise ValueError(f"Custom filter '{filter_name}' not found")
        
        curve = cls._custom_filters[filter_name]
        curve.save_to_file(filepath)
        print(f"Saved filter '{filter_name}' to: {filepath}")
    
    @classmethod
    def list_filter_info(cls):
        """Print information about all filters."""
        print(f"Default broadband filters ({len(cls._default_broadband)}): {', '.join(cls._default_broadband)}")
        print(f"Default mediumband filters ({len(cls._default_mediumband)}): {len(cls._default_mediumband)} filters")
        print(f"Custom filters ({len(cls._custom_filters)}):")
        
        for name, curve in cls._custom_filters.items():
            wl_range = f"{curve.wavelength.min():.0f}-{curve.wavelength.max():.0f} Å"
            print(f"  {name}: {len(curve.wavelength)} points, {wl_range}, source: {curve.source_type}")
    
    @classmethod
    def clear_custom_filters(cls):
        """Remove all custom filters."""
        count = len(cls._custom_filters)
        cls._custom_filters.clear()
        print(f"Cleared {count} custom filters")
        
    
    @classmethod
    def _query_facilities_and_instruments(cls):
        SvoFps.get_filter_index(1e-4*u.um, 600*u.um, timeout=12000)
        
        facilities = {k: set(g["Instrument"].dropna().tolist()) for k, g in index.to_pandas().groupby("Facility")}
        
        for facility, instruments in facilities.items():
            instruments.discard("")
            if not instruments:
                try:
                    response = SvoFps.get_filter_list(facility=facility)
                    facilities[facility] = set(response.to_pandas()['filterID'].unique())
                except Exception as e:
                    print(f"Failed to query instruments for facility '{facility}': {e}")
        
        facilities = {key: list(value) for key, value in facilities.items()}
        
        return facilities    
    
    
    @classmethod
    def get_catcols(cls, cat_type, col_names):
        """Return a dictionary of given type"""
        catcols ={"cigale": cls.cigale,
                  "eazy": cls.eazy,
                  "lephare": cls.lephare,
                  "ppxf": cls.ppxf,
                  "goyangyi": cls.goyangyi
                  }
        
        function = catcols[cat_type.lower()]
        sig = inspect.signature(function)
        
        # image_data, header, error_data, galaxy_name, observatory, band, image_set
        kwargs = {"self": cls, "col_names": col_names}
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return function(**filtered_kwargs)
    
    
    def cigale(self):
        cols_cigale = {
            'GALEX.NUV': 'galex.NUV',
            'GALEX.FUV': 'galex.FUV',
            'SDSS.u': 'sloan.sdss.u',
            'SDSS.g': 'sloan.sdss.g',
            'SDSS.r': 'sloan.sdss.r',
            'SDSS.i': 'sloan.sdss.i',
            'SDSS.z': 'sloan.sdss.z',
            'PanStarrs.y': 'PAN-STARRS_y',
            '2MASS.J': 'J_2mass',
            '2MASS.H': 'H_2mass',
            '2MASS.Ks': 'Ks_2mass',
            'Spitzer.ch1': 'spitzer.irac.ch1',
            'Spitzer.ch2': 'spitzer.irac.ch2',
            'Spitzer.ch3': 'spitzer.irac.ch3',
            'Spitzer.ch4': 'spitzer.irac.ch4',
            'WISE.w1': 'wise.W1',
            'WISE.w2': 'wise.W2',
            'WISE.w3': 'wise.W3',
            'WISE.w4': 'wise.W4',
            'F657N': 'HST.UVIS1.F657N',
            'F658N': 'HST.UVIS1.F658N',
        }
        cols_cigale.update({f'{key}_err': f'{cols_cigale[key]}_err' for key in cols_cigale.keys() if '_err' not in key})
        return cols_cigale
    
    def eazy(self, col_names):
        flux_dict = {name:f"F_{name}" for name in col_names if "_err" not in name}
        err_dict = {name:f"E_{name.strip('_err')}" for name in col_names if "_err" in name}
        flux_dict.update(err_dict)
        
        cols_eazy = flux_dict
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
        cols_cigale = self.cigale(self)
        print(" ╱|、\n(˚ˎ 。7  \n |、˜〵          \n じしˍ,)ノ")
        return cols_cigale