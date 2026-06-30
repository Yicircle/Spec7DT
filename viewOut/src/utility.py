import re
import numpy as np
import astropy.units as u
from astropy.table import Table
from astroquery.simbad import Simbad

try:
    from .filter_property import CurveProp, Filters, Observatories
except ImportError:
    from filter_property import CurveProp, Filters, Observatories


class GalaxyUtils:
    def query_distance(
        target: str
    ) -> float:
        """
        Query SIMBAD for the distance of a target.
        Returns distance in parsecs.
        """
        try:
            Simbad.add_votable_fields('mesDistance')

            result_table = Simbad.query_object(target).to_pandas()
            result_table['mesdistance.method'] = result_table['mesdistance.method'].str.strip(' ')
            try:
                result_table = result_table[np.logical_or(result_table['mesdistance.method'] == 'Ceph', result_table['mesdistance.method'] == 'Cep')]
            except:
                print("No Cepheid distance found, using all methods.")
            
            mean_error = (np.abs(result_table['mesdistance.minus_err']) + np.abs(result_table['mesdistance.plus_err'])) / 2

            d_Mpc = result_table['mesdistance.dist'][mean_error.idxmin()] * u.Mpc
            
        except:
            simbad = Simbad()
            simbad.add_votable_fields("distance")
            result = simbad.query_object(target)
            result = result[result["Distance_unit"] == "Mpc"]
            d_Mpc = result["Distance_distance"].value[0] * u.Mpc
        
        return d_Mpc.value

    def redshift_to_distance(
        redshift: float
    ) -> float:
        """
        Convert redshift to distance in megaparsecs.
        Uses the cosmological parameters from astropy.
        """
        from astropy.cosmology import Planck18 as cosmo
        return cosmo.luminosity_distance(redshift).to(u.Mpc).value
    


class UnitConverter:
    """
    Flux [mJy] -> Luminosity [W], magnitude [mag]
    Luminosity [W] -> Luminosity [L_sol]
    Mass [M_sol] -> Mass [kg]
    Metallicity [Z_sol] -> Metallicity [Z/H]
    Age [Myr] -> log Age [log yr], Age [yr]
    
    """
    
    def from_flux(self, flux, flux_err, to="W", D_m=1.0):
        if (to == "W") or (to.lower() == "lum") or (to.lower == "luminosity"):
            L_factor = 4 * np.pi * D_m**2 * 1e-29 * 8.98297413888032 * 1e13  # only for bessell V filter
            return L_factor * flux, L_factor * flux_err
            
        elif (to.lower() == "mag") or (to.lower() == "magnitude"):
            return -2.5 * np.log10(flux / 3.631e6), np.abs(-2.5 * flux_err / (flux * np.log(10)))
        
        else:
            raise ValueError("Unknown Unit.")
        
    def from_luminosity(self, lum, lum_err, to="L_sol"):
        if (to == "L_sol") or (to == "solor"):
            return lum / 3.84e26, lum_err / 3.84e26
        
        else:
            raise ValueError("Unknown Unit.")
        
    def from_metallicity(self, metal, metal_err, to="ZtoH", Z_sol=0.02):
        if (to == "ZtoH") or (to == "Z/H"):
            return np.log10(metal / Z_sol), np.abs(metal_err / (metal * np.log(10)))
        
    def from_age(self, age, age_err, to="log", start="Myr"):
        if start == "Myr":
            if to == "log":
                return np.log10(age * 1e6), np.abs(age_err * 1e6 / (age * 1e6 * np.log(10)))
            elif to == "Gyr":
                return age * 1e-3, age_err * 1e-3
            elif to == "yr":
                return age * 1e6, age_err * 1e6
        elif start == "yr":
            if to == "log":
                return np.log10(age), np.abs(age_err / (age * np.log(10)))
            elif to == "Gyr":
                return age * 1e-9, age_err * 1e-9
            elif to == "Myr":
                return age * 1e-6, age_err * 1e-6
            

class TypeParser:
    def __init__(self):
        pass
    
    @classmethod
    def parse_type(cls, property_name, unit_dict: dict=None):
        astropy_unit_types = {
            "spectral flux density": "flux",
            "power": "luminosity",
            "time": "age",
            "mass": "mass",
            "mag": "mag"
        }
        
        unit_types = {
            "age": "age",
            ".m_": "mass",
            "lum": "luminosity",
            "mass": "mass",
            "metal": "metallicity",
            "attenuation": "attenuation",
            "sfr": "sfr"
        }
        
        def get_type_from_name(_name):
            return next((v for k, v in unit_types.items() if k in _name), "unknown")
        
        
        if unit_dict is not None: 
            if unit_dict[property_name] is not None:
                ptype = unit_dict[property_name].physical_type
            
                if ptype == "unknown":
                    ptype = str(unit_dict[property_name])
                    
                return astropy_unit_types.get(str(ptype), get_type_from_name(property_name))
            else:
                return get_type_from_name(property_name)

        else:
            return get_type_from_name(property_name)


class Parsers:
    filter_inst = Filters()
    
    def __init__(self):
        pass
    
    @staticmethod
    def _observatory_name_parser(file_name):
        """Parse observatory names from a given string or list."""
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")
        for obs in Observatories.get_observatories():
            pattern = "|".join(map(re.escape, ['-', ' ', '_', '.', '/']))
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
            pattern = "|".join(map(re.escape, ['-', ' ', '_', '.', '/']))
            if band in re.split(pattern, file_name):
                return band
        return None
    

    @classmethod
    def get_filter_properties(cls, filter_list):
        obses = []
        filters = []
        filter_inst = Filters()
        
        for _name in filter_list:
            obses.append(cls._observatory_name_parser(_name))
            filters.append(cls._band_name_parser(_name, filter_inst))
        
        methods = {
            'pivot': CurveProp.pivot_wavelength,
            'mean': CurveProp.mean_wavelength,
            'peak': CurveProp.peak_wavelength,
            'center': CurveProp.center_wavelength,
            'FWHM': CurveProp.FWHM
        }
        
        resp_dic = {
            f"{obs}.{filt}": filter_inst.get_filter_curve(name=filt, facility=obs)
            for obs, filt in zip(obses, filters)
        }

        resp_props = {
            method_name: {key: func(wave, resp) for key, (wave, resp) in resp_dic.items()}
            for method_name, func in methods.items()
        }
        
        return resp_props, resp_dic
    
    @classmethod
    def get_filter_inst(cls):
        return cls.filter_inst
