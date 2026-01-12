from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Tuple
import numpy as np
from astroquery.svo_fps import SvoFps
from astropy import units as u
from importlib import resources
import inspect

from ..utils.utility import Observatories

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
    """Manages astronomical filter transmission curves from multiple sources."""
    _filters: Dict[str, FilterCurve] = {}
    _index = SvoFps.get_filter_index(1e-4*u.um, 600*u.um, timeout=12000).to_pandas()
    
    def __init__(self):
        """Initialize and auto-load predefined filters from package resources."""
        self._load_predefined_filters()
    
    @classmethod
    def _load_predefined_filters(cls):
        """Load all .dat files from package filter_curves directory."""
        try:
            # from files
            filter_dir = resources.files("Spec7DT.reference.filter_curves")
            for filepath in filter_dir.iterdir():
                if filepath.name.endswith('.dat'):
                    try:
                        # Read file content directly using as_file context manager
                        with resources.as_file(filepath) as file_path:
                            cls._load_from_file(cls, file_path)
                    except Exception as e:
                        print(f"Warning: Failed to load {filepath.name}: {str(e)}")
            
            # from SVO
            
            obs = Observatories.get_observatories()
            mask_index = cls._index[["Facility", "Instrument"]].isin(obs)
            mask_index = (mask_index["Facility"] | mask_index["Instrument"])
            filterIDs = cls._index[mask_index]["filterID"].to_numpy()
            
            for filter_name in filterIDs:
                cls._load_from_svo(cls, filter_name)
            print("All Filters loaded")
                
        except Exception as e:
            print(f"Warning: Could not load predefined filters: {str(e)}")
    
    def _load_from_file(self, filepath: Union[str, Path], name: Optional[str] = None):
        """
        Protected method: Load filter from ASCII file.
        Expected format: header lines starting with '#', then wavelength-response columns.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Filter file not found: {filepath}")
        
        # Read file and parse header/data
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header (lines starting with #)
        header = [line.strip('# \n') for line in lines if line.startswith('#')]
        filter_name = header[0] if len(header) > 0 else filepath.stem
        unit_type = header[1] if len(header) > 1 else 'photon'
        description = header[2] if len(header) > 2 else ''
        
        # Parse data (non-comment lines)
        data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
        data = np.array([list(map(float, line.split())) for line in data_lines])
        
        wavelength = data[:, 0]
        response = data[:, 1]
        
        # Create FilterCurve object
        if name:
            filter_name = name
        
        curve = FilterCurve(
            name=filter_name,
            wavelength=wavelength,
            response=response,
            source_type='file',
            source_path=str(filepath),
            unit_type=unit_type,
            description=description
        )
         
        self._filters[filter_name] = curve
    
    def _load_from_svo(self, filter_id: str, name: Optional[str] = None):
        """
        Protected method: Load filter from SVO Filter Profile Service.
        
        Args:
            filter_id: SVO filter identifier (e.g., 'SLOAN/SDSS.u', '2MASS/2MASS.J')
            name: Optional custom name. If None, uses filter_id as name
        """
        try:
            # Query SVO for filter transmission data
            data = SvoFps.get_transmission_data(filter_id)
            
            # Extract wavelength and transmission
            wavelength = data['Wavelength'].to('angstrom').value
            transmission = data['Transmission'].value
            
            # Get filter metadata
            facility_name = filter_id.split("/")[0]
            meta = SvoFps.get_filter_list(facility=facility_name).to_pandas()
            meta = meta[meta['filterID'] == filter_id].reset_index(drop=True)
            description = f"{meta.loc[0, 'Description']}"
            unit_type = "photon" if int(meta.loc[0, 'DetectorType']) else "energy"
            
            # Create FilterCurve object
            filter_name = name if name else filter_id
            curve = FilterCurve(
                name=filter_name,
                wavelength=wavelength,
                response=transmission,
                source_type='default',
                source_path="SVO Filter Service Web",
                unit_type=unit_type,
                description=description
            )
            
            self._filters[filter_name] = curve
            
        except Exception as e:
            raise ValueError(f"Failed to load SVO filter '{filter_id}': {str(e)}")
    
    def _search_svo_filter(self, facility: Optional[str] = None, 
                          instrument: Optional[str] = None, 
                          filter_name: Optional[str] = None) -> str:
        """
        Search SVO database for filter ID using flexible naming.
        
        Args:
            facility: Facility/Observatory name (case-insensitive)
            instrument: Instrument name (case-insensitive)
            filter_name: Filter band name (case-insensitive)
            
        Returns:
            Best matching SVO filter ID
        """
        
        # Build query parameters (SVO accepts these fields)
        query_params = {}
        if facility:
            query_params['facility'] = facility
        if instrument:
            query_params['instrument'] = instrument
        
        # Query SVO filter list
        try:
            filter_list = SvoFps.get_filter_list(**query_params)
        except IndexError:
            filter_list = SvoFps.get_filter_list(facility=query_params["facility"])
        except Exception as e:
            raise ValueError(f"SVO query failed: {str(e)}")
        
        if len(filter_list) == 0:
            raise ValueError(f"No filters found for {query_params}")
        
        # If filter_name specified, search for it in filterID field
        if filter_name:
            filter_name_lower = filter_name.lower()
            matches = []
            
            for row in filter_list:
                filter_id = row['filterID']
                # Extract last part after final dot or slash
                band = filter_id.split('.')[-1].split('/')[-1].lower()
                
                if band == filter_name_lower or filter_name_lower in band:
                    matches.append(filter_id)
            
            if len(matches) == 0:
                raise ValueError(f"No filter matching '{filter_name}' found in {len(filter_list)} results")
            elif len(matches) > 1:
                exact_match = [_filter_id for _filter_id in matches if filter_name_lower == _filter_id.split('.')[-1].split('/')[-1].lower()]
                if len(exact_match) == 1:
                    matches = exact_match
                else:
                    pass
            
            return matches[0]
        else:
            # No filter name - return first result
            return filter_list[0]['filterID']
    
    @classmethod
    def load_filter(cls, source: Union[str, Path], name: Optional[str] = None,
                   facility: Optional[str] = None, instrument: Optional[str] = None,
                   filter_name: Optional[str] = None):
        """
        Load filter from file or SVO service.
        
        For files:
            load_filter('path/to/filter.dat')
            load_filter('custom_filter.dat', name='my_filter')
        
        For SVO (flexible naming):
            load_filter('svo', facility='SLOAN', instrument='SDSS', filter_name='u')
            load_filter('svo', facility='2MASS', filter_name='J')
            load_filter('svo', instrument='WFC3', filter_name='F606W')
        
        Args:
            source: File path or 'svo' for SVO service
            name: Optional custom name for the filter
            facility: Facility name (for SVO, case-insensitive)
            instrument: Instrument name (for SVO, case-insensitive)
            filter_name: Filter band name (for SVO, case-insensitive)
        """
        source_str = str(source).lower()
        
        if source_str == 'svo':
            # SVO service - search for filter ID
            if not any([facility, instrument, filter_name]):
                raise ValueError("Must provide at least one of: facility, instrument, or filter_name")
            
            filter_id = cls._search_svo_filter(cls, facility, instrument, filter_name)
            
            # Use custom name if provided, otherwise use filter_name or filter_id
            if name:
                final_name = name
            else:
                final_name = filter_id
            
            cls._load_from_svo(cls, filter_id, final_name)
        else:
            # File path
            cls._load_from_file(cls, str(source))
    
    @classmethod
    def add_custom(cls, name: str, wavelength: np.ndarray, response: np.ndarray, 
                   unit_type: str = 'photon', description: str = ''):
        """
        Add custom filter from arrays.
        
        Args:
            name: Filter name
            wavelength: Wavelength array (Angstroms)
            response: Transmission/response array
            unit_type: 'photon' or 'energy'
            description: Optional description
        """
        curve = FilterCurve(
            name=name,
            wavelength=wavelength,
            response=response,
            source_type='array',
            source_path=None,
            unit_type=unit_type,
            description=description
        )
        
        cls._filters[name] = curve
        print(f"Added custom filter: {name}")
    
    @classmethod
    def get_filter_curve(cls, name: str = None, 
                    facility: Optional[str] = None, 
                    instrument: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get wavelength and transmission arrays for a filter.
        
        Args:
            name: Filter name
            
        Returns:
            Tuple of (wavelength, response) arrays
        """
        curve = cls.get_filter(name=name, facility=facility, instrument=instrument)
        return curve.wavelength, curve.response
    
    @classmethod
    def get_filter(cls, name: str = None, 
                    facility: Optional[str] = None, 
                    instrument: Optional[str] = None) -> FilterCurve:
        """
        Get complete FilterCurve object for a filter.
        
        Args:
            name: Filter name
            
        Returns:
            FilterCurve object
        """
        if (facility is not None) and (instrument is not None):    
            filterID = f"{facility}/{instrument}.{name}".lower()
        elif facility is not None:
            lower_keys = [key.lower() for key in set(cls._filters.keys())]
            match_list = [key for key in lower_keys if (facility.lower() in key) and (name.lower() in key)]
            filterID = match_list[0] if len(match_list) > 0 else name
        elif instrument is not None:
            lower_keys = [key.lower() for key in set(cls._filters.keys())]
            match_list = [key for key in lower_keys if (instrument.lower() in key) and (name.lower() in key)]
            filterID = match_list[0] if len(match_list) > 0 else name
            
            
        if name in cls._filters:
            curve = cls._filters[name]
            return curve
        
        elif filterID in (key.lower() for key in cls._filters.keys()):
            curve = cls._filters[next(key for key in cls._filters.keys() if key.lower() == filterID)]
            return curve
        else:
            raise KeyError(f"Filter '{name}' not found. Available: {cls.list_filters(cls)}")
    
    @classmethod
    def get_all_filters(cls):
        return [_filter.split("/")[-1].split(".")[-1] for _filter in cls.list_filters(cls)]
    
    def list_filters(self) -> list:
        """Return list of available filter names."""
        return list(self._filters.keys())
    
    def __getitem__(self, name: str) -> FilterCurve:
        """Allow dictionary-style access to filters."""
        return self.get_filter_curve(name)
    
    def __contains__(self, name: str) -> bool:
        """Check if filter exists."""
        return name in self._filters
    
    def __len__(self) -> int:
        """Return number of loaded filters."""
        return len(self._filters)
    
    def __repr__(self) -> str:
        """String representation showing loaded filters."""
        return f"Filter({len(self)} filters: {', '.join(self.list_filters())})"
    
    
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
            'PACS.blue': 'herschel.pacs.blue',
            'PACS.green': 'herschel.pacs.green',
            'PACS.red': 'herschel.pacs.red',
            'SPIRE.PSW': 'herschel.spire.PSW',
            'SPIRE.PMW': 'herschel.spire.PMW',
            'SPIRE.PLW': 'herschel.spire.PLW',
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