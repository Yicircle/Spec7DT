from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Tuple
import numpy as np
from importlib import resources
import inspect

from .filter_properties import (
    FilterProperties,
    FilterPropertyCalculator,
    prepare_filter_curve,
)


def _get_svo_fps():
    """Import SVO only when a network-backed filter lookup is requested."""
    try:
        from astroquery.svo_fps import SvoFps
    except ImportError as exc:
        raise ImportError("astroquery is required to load filters from SVO") from exc
    return SvoFps


@dataclass
class FilterCurve:
    """Container for filter response curve data."""
    name: str
    wavelength: np.ndarray
    response: np.ndarray
    source_type: str  # 'default', 'file', 'array', or 'svo'
    source_path: Optional[str] = None
    unit_type: str = 'photon'  # 'photon' or 'energy'
    description: str = ''
    
    def __post_init__(self):
        """Validate and convert arrays to numpy."""
        self.wavelength, self.response = prepare_filter_curve(
            self.wavelength,
            self.response,
            require_positive_response=False,
        )

        if self.unit_type not in {'photon', 'energy'}:
            raise ValueError("unit_type must be either 'photon' or 'energy'")

    def get_properties(self) -> FilterProperties:
        """Calculate all supported properties for this filter curve."""
        return FilterPropertyCalculator.calculate(self.wavelength, self.response)

    @property
    def properties(self) -> FilterProperties:
        """Calculated filter response properties."""
        return self.get_properties()

    @property
    def pivot_wavelength(self) -> float:
        """Pivot wavelength of the response curve."""
        return FilterPropertyCalculator.pivot_wavelength(self.wavelength, self.response)

    @property
    def mean_wavelength(self) -> float:
        """Response-weighted mean wavelength."""
        return FilterPropertyCalculator.mean_wavelength(self.wavelength, self.response)

    @property
    def peak_wavelength(self) -> float:
        """Wavelength at maximum response."""
        return FilterPropertyCalculator.peak_wavelength(self.wavelength, self.response)

    @property
    def center_wavelength(self) -> float:
        """Center wavelength based on half-maximum crossings."""
        return FilterPropertyCalculator.center_wavelength(self.wavelength, self.response)

    @property
    def fwhm(self) -> float:
        """Full width at half maximum."""
        return FilterPropertyCalculator.fwhm(self.wavelength, self.response)
    
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
    _predefined_loaded: bool = False
    
    def __init__(self):
        """Initialize and load package-bundled filter curves."""
        self._load_predefined_filters()
    
    @classmethod
    def _load_predefined_filters(cls):
        """Load all .dat files from package filter_curves directory."""
        if cls._predefined_loaded:
            return

        try:
            filter_dir = resources.files("Spec7DT.reference.filter_curves")
            for filepath in filter_dir.iterdir():
                if filepath.name.endswith('.dat'):
                    try:
                        with resources.as_file(filepath) as file_path:
                            cls._load_from_file(file_path, source_type='default')
                    except Exception as e:
                        print(f"Warning: Failed to load {filepath.name}: {str(e)}")
            cls._predefined_loaded = True
        except Exception as e:
            print(f"Warning: Could not load predefined filters: {str(e)}")
    
    @classmethod
    def _load_from_file(
        cls,
        filepath: Union[str, Path],
        name: Optional[str] = None,
        source_type: str = 'file',
    ):
        """
        Protected method: Load filter from ASCII file.
        Expected format: header lines starting with '#', then wavelength-response columns.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Filter file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header = [line[1:].strip() for line in lines if line.startswith('#')]
        filter_name = header[0] if len(header) > 0 else filepath.stem
        unit_type = header[1].lower() if len(header) > 1 else 'photon'
        description = header[2] if len(header) > 2 else ''
        
        data = np.loadtxt(filepath, comments='#', ndmin=2)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Filter file must contain wavelength and response columns")
        
        wavelength = data[:, 0]
        response = data[:, 1]
        
        if name:
            filter_name = name
        
        curve = FilterCurve(
            name=filter_name,
            wavelength=wavelength,
            response=response,
            source_type=source_type,
            source_path=str(filepath),
            unit_type=unit_type,
            description=description
        )
         
        cls._filters[filter_name] = curve
    
    @classmethod
    def _load_from_svo(cls, filter_id: str, name: Optional[str] = None):
        """
        Protected method: Load filter from SVO Filter Profile Service.
        
        Args:
            filter_id: SVO filter identifier (e.g., 'SLOAN/SDSS.u', '2MASS/2MASS.J')
            name: Optional custom name. If None, uses filter_id as name
        """
        SvoFps = _get_svo_fps()
        try:
            data = SvoFps.get_transmission_data(filter_id)
            wavelength = data['Wavelength'].to('angstrom').value
            transmission = data['Transmission'].value

            facility_name = filter_id.split("/")[0]
            description = ''
            unit_type = 'photon'

            try:
                meta = SvoFps.get_filter_list(facility=facility_name).to_pandas()
                meta = meta[meta['filterID'] == filter_id].reset_index(drop=True)
                if len(meta) > 0:
                    description = f"{meta.loc[0, 'Description']}"
                    unit_type = "photon" if int(meta.loc[0, 'DetectorType']) else "energy"
            except Exception:
                pass

            filter_name = name if name else filter_id
            curve = FilterCurve(
                name=filter_name,
                wavelength=wavelength,
                response=transmission,
                source_type='svo',
                source_path="SVO Filter Service Web",
                unit_type=unit_type,
                description=description
            )
            
            cls._filters[filter_name] = curve
            
        except Exception as e:
            raise ValueError(f"Failed to load SVO filter '{filter_id}': {str(e)}")
    
    @classmethod
    def _search_svo_filter(cls, facility: Optional[str] = None,
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
        
        SvoFps = _get_svo_fps()
        try:
            filter_list = SvoFps.get_filter_list(**query_params)
        except IndexError:
            filter_list = SvoFps.get_filter_list(facility=query_params["facility"])
        except Exception as e:
            raise ValueError(f"SVO query failed: {str(e)}")
        
        if len(filter_list) == 0:
            raise ValueError(f"No filters found for {query_params}")
        
        if filter_name:
            filter_name_lower = filter_name.lower()
            matches = []
            
            for row in filter_list:
                filter_id = row['filterID']
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
            if not any([facility, instrument, filter_name]):
                raise ValueError("Must provide at least one of: facility, instrument, or filter_name")
            
            filter_id = cls._search_svo_filter(facility, instrument, filter_name)
            
            if name:
                final_name = name
            else:
                final_name = filter_id
            
            cls._load_from_svo(filter_id, final_name)
        else:
            cls._load_from_file(str(source), name=name)
    
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
    def _matching_filter_keys(cls, name: str,
                              facility: Optional[str] = None,
                              instrument: Optional[str] = None) -> list:
        """Return filter keys matching a flexible name query."""
        cls._load_predefined_filters()
        name_lower = name.lower()
        keys = list(cls._filters.keys())

        exact = [key for key in keys if key.lower() == name_lower]
        if exact:
            return exact

        candidates = []
        if facility and instrument:
            candidates.append(f"{facility}/{instrument}.{name}".lower())
            candidates.append(f"{facility}.{instrument}.{name}".lower())
        if facility:
            candidates.append(f"{facility}.{name}".lower())
        if instrument:
            candidates.append(f"{instrument}.{name}".lower())

        for candidate in candidates:
            matches = [key for key in keys if key.lower() == candidate]
            if matches:
                return matches

        matches = []
        for key in keys:
            key_lower = key.lower()
            band = key_lower.split('/')[-1].split('.')[-1]
            if band != name_lower:
                continue
            if facility and facility.lower() not in key_lower:
                continue
            if instrument and instrument.lower() not in key_lower:
                continue
            matches.append(key)

        return matches
    
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
    def get_filter_properties(cls, name: str = None,
                              facility: Optional[str] = None,
                              instrument: Optional[str] = None) -> FilterProperties:
        """Get calculated response-curve properties for a filter."""
        return cls.get_filter(name=name, facility=facility, instrument=instrument).get_properties()
    
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
        if name is None:
            raise ValueError("Filter name must be provided")

        matches = cls._matching_filter_keys(name, facility=facility, instrument=instrument)
        if len(matches) == 1:
            return cls._filters[matches[0]]
        if len(matches) > 1:
            raise KeyError(f"Filter '{name}' is ambiguous. Matches: {matches}")

        raise KeyError(f"Filter '{name}' not found. Available: {cls.list_filters()}")
    
    @classmethod
    def get_all_filters(cls):
        return [_filter.split("/")[-1].split(".")[-1] for _filter in cls.list_filters()]
    
    @classmethod
    def list_filters(cls) -> list:
        """Return list of available filter names."""
        cls._load_predefined_filters()
        return list(cls._filters.keys())
    
    def __getitem__(self, name: str) -> FilterCurve:
        """Allow dictionary-style access to filters."""
        return self.get_filter(name)
    
    def __contains__(self, name: str) -> bool:
        """Check if filter exists."""
        try:
            self.get_filter(name)
            return True
        except (KeyError, ValueError):
            return False
    
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
            'WISE.W1': 'wise.W1',
            'WISE.W2': 'wise.W2',
            'WISE.W3': 'wise.W3',
            'WISE.W4': 'wise.W4',
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
