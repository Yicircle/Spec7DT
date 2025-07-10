import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class conversion:
    def __init__(self):
        self.obs_match = {'GALEX': self.GALEX,
                         'SDSS': self.SDSS,
                         '7DT': self.SDT,
                         'PS1': self.PanStarr1,
                         'Spitzer': self.Spitzer,
                         'WISE': self.WISE,
                         '2MASS': self.TMASS
                         }
    
    def unitConvertor(self, image_data, header, error_data, galaxy_name, observatory, band, image_set):
        """
        Convert image data to consistent flux units based on observatory and band.
        
        Parameters:
        -----------
        image_data : array-like
            Raw image data to be converted
        header : dict
            FITS header or dictionary containing metadata
        galaxy_name : str
            Name of the galaxy (for logging/tracking purposes)
        observatory : str
            Observatory name (key in obs_match)
        band : str
            Filter/band name
            
        Returns:
        --------
        array-like
            Converted flux data in mJy
        """
        
        # Check if observatory is supported
        if observatory not in self.obs_match:
            raise ValueError(f"Observatory '{observatory}' not supported. "
                           f"Available observatories: {list(self.obs_match.keys())}")
        
        # Get the conversion function
        conversion_func = self.obs_match[observatory]
        
        try:
            # All functions now have standardized signature
            convert_im, convert_err = conversion_func(image_data, header, band), conversion_func(error_data, header, band)
            image_set.update_data(convert_im, galaxy_name, observatory, band)
            image_set.update_error(convert_err, galaxy_name, observatory, band)
                
        except Exception as e:
            print(f"Error converting data for {galaxy_name} from {observatory} {band}: {str(e)}")
            raise
    
    # UVs
    def GALEX(self, image_data, header=None, band=None):
        return np.array(image_data) * 1e-3
    
    
    # Opticals
    def SDSS(self, image_data, header=None, band=None):  # nmgy
        return np.array(image_data) * 3.631 * 10 ** (-3)
    
    def SDT(self, image_data, header, band=None):  # ADU
        try:
            ZP = header['ZP_AUTO']
        except KeyError:
            raise KeyError("Zero Point value is not found.")
        flux = 3631 * (image_data) * 10 ** (-ZP / 2.5) * 1e3
        return flux

    def PanStarr1(self, image_data, header, band=None):
        exptime = header['EXPTIME']
        mag = -2.5 * np.log10(image_data) + 25 + 2.5 * np.log10(exptime)
        flux = 10 ** (-0.4 * mag) * 3631 * 1e3
        return flux
    
    
    # IRs
    def Spitzer(self, image_data, header=None, band=None):  # MJy/sr
        return 1e9 * ((1.1e-4 * 3600 / 206265.0) ** 2) * image_data
    
    def WISE(self, image_data, header=None, band=None):  #ADU/DN
        # WISE flux calibration factors
        F_nu0 = {'w1': 309.540,
                'w2': 171.787,
                'w3': 31.674,
                'w4': 8.363
                }
        M_inst0 = {'w1': 20.5,
                'w2': 19.5,
                'w3': 18.0,
                'w4': 13.0
                }

        image_data = np.where(image_data == 0, 0.1, image_data)
        image_data = np.where(np.isnan(image_data), 0.1, image_data)
        M_cal = M_inst0[band] - 2.5 * np.log10(image_data)
        flux = np.power(10, - M_cal / 2.5) * F_nu0[band] * 1e3
        return flux
    
    def TMASS(self, image_data, header, band=None):
        zp_2mass = {'j': 21.1258,
                    'h': 20.7288,
                    'k': 20.1106
                    }
        zpflux_2mass = {'j': 1594,
                        'h': 1024,
                        'k': 666.7
                        }

        mag = -2.5 * np.log10(image_data) + zp_2mass[header['FILTER']]
        f_nu = 10 ** (-0.4 * mag) * zpflux_2mass[header['FILTER']] * 1e3  # [mJy]
        return f_nu