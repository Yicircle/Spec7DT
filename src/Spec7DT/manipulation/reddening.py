import numpy as np
import warnings

from ..handlers.filter_handler import Filters
from ..utils.utility import useful_functions

class Reddening:
    def __init__(self):
        try:
            from dustmaps.config import config
            config.reset()
        except Exception:
            pass
        self.filter_inst = Filters()
    
    def dered(self, image_data, header, error_data, galaxy_name, observatory, band, image_set,
              metadata_resolver=None, filter_config=None):
        self.obj, self.obs, self.filt = galaxy_name, observatory, band
        self.filter_config = filter_config or {}
        coords = useful_functions.get_sky_loc(
            self.obj,
            header=header,
            metadata_resolver=metadata_resolver,
            required=False,
        )
        if coords is None:
            warnings.warn(f"Skipping dereddening for {self.obj}: galaxy coordinate unavailable.")
            return

        try:
            from dustmaps.planck import PlanckQuery

            planck = PlanckQuery()
            self.ebv = planck(coords)
        except Exception as exc:
            warnings.warn(f"Skipping dereddening for {self.obj}: dust map query failed ({exc}).")
            return
        
        try:
            wave, resp = self.get_resp_curve()
        except Exception as exc:
            warnings.warn(
                f"Skipping dereddening for {self.obj}: filter curve unavailable ({exc})."
            )
            return
        
        # Check if filter is valid for CCM98 model
        if (max(wave) > 3.3 * 1e4) | (min(wave) < 9.1 * 1e2):
            print('Filter is not valid for CCM98 model')
            return 0
        
        A_mean = self._calculate_mean_extinction(wave, resp)
        deredden_img = image_data * 10 ** (0.4 * A_mean)
        deredden_err = error_data * 10 ** (0.4 * A_mean)
        
        image_set.update_data(deredden_img, galaxy_name, observatory, band)
        image_set.update_error(deredden_err, galaxy_name, observatory, band)
        
                
    def get_resp_curve(self, verbose=False):
        """Get response curve using the Filters class."""
        
        try:
            # Try to get filter curve using observatory and band
            curve = self.filter_inst.ensure_filter(
                name=self.filt,
                facility=self.obs,
                **getattr(self, "filter_config", {}),
            )
            
            # Filter out zero response values
            mask = (curve.response != 0)
            wave = curve.wavelength[mask]
            resp = curve.response[mask]
            
            if verbose:
                print(f"Loaded filter: {curve.name} ({curve.unit_type})")
                if curve.description:
                    print(f"Description: {curve.description}")
            
            return wave, resp
            
        except Exception:
            raise
    
    def reddening_ccm(self, wave, ebv=None, a_v=None, r_v=3.1, model='ccm89'):
        """
        Not used in FIREFLY
        Determines a CCM reddening curve.

        Parameters
        ----------
        wave: ~numpy.ndarray
            wavelength in Angstroms
        flux: ~numpy.ndarray
        ebv: float
            E(B-V) differential extinction; specify either this or a_v.
        a_v: float
            A(V) extinction; specify either this or ebv.
        r_v: float, optional
            defaults to standard Milky Way average of 3.1
        model: {'ccm89', 'gcc09'}, optional
            * 'ccm89' is the default Cardelli, Clayton, & Mathis (1989) [1]_, but
            does include the O'Donnell (1994) parameters to match IDL astrolib.
            * 'gcc09' is Gordon, Cartledge, & Clayton (2009) [2]_. This paper has
            incorrect parameters for the 2175A bump; not yet corrected here.

        Returns
        -------
        reddening_curve: ~numpy.ndarray
            Multiply to deredden flux, divide to redden.

        Notes
        -----
        Cardelli, Clayton, & Mathis (1989) [1]_ parameterization is used for all
        models. The default parameter values are from CCM except in the optical
        range, where the updated parameters of O'Donnell (1994) [3]_ are used
        (matching the Goddard IDL astrolib routine CCM_UNRED).

        The function is works between 910 A and 3.3 microns, although note the
        default ccm89 model is scientifically valid only at >1250 A.

        Model gcc09 uses the updated UV coefficients of Gordon, Cartledge, & Clayton
        (2009) [2]_, and is valid from 910 A to 3030 A. This function will use CCM89
        at longer wavelengths if GCC09 is selected, but note that the two do not
        connect perfectly smoothly. There is a small discontinuity at 3030 A. Note
        that GCC09 equations 14 and 15 apply to all x>5.9 (the GCC09 paper
        mistakenly states they do not apply at x>8; K. Gordon, priv. comm.).

        References
        ----------
        [1] Cardelli, J. A., Clayton, G. C., & Mathis, J. S. 1989, ApJ, 345, 245
        [2] Gordon, K. D., Cartledge, S., & Clayton, G. C. 2009, ApJ, 705, 1320
        [3] O'Donnell, J. E. 1994, ApJ, 422, 158O

        """

        import warnings

        model = model.lower()
        if model not in ['ccm89','gcc09']:
            raise ValueError('model must be ccm89 or gcc09')
        if (a_v is None) and (ebv is None):
            raise ValueError('Must specify either a_v or ebv')
        if (a_v is not None) and (ebv is not None):
            raise ValueError('Cannot specify both a_v and ebv')
        if a_v is not None:
            ebv = a_v / r_v

        if model == 'gcc09':
            raise ValueError('TEMPORARY: gcc09 currently does 2175A bump '+
                'incorrectly')

        x = 1e4 / wave      # inverse microns
        if any(x < 0.3) or any(x > 11):
            raise ValueError('ccm_dered valid only for wavelengths from 910 A to '+
                '3.3 microns')
        if any(x > 8) and (model == 'ccm89'):
            warnings.warn('CCM89 should not be used below 1250 A.')
        #    if any(x < 3.3) and any(x > 3.3) and (model == 'gcc09'):
        #        warnings.warn('GCC09 has a discontinuity at 3030 A.')

        a = np.zeros(x.size)
        b = np.zeros(x.size)

        # NIR
        valid = (0.3 <= x) & (x < 1.1)
        a[valid] = 0.574 * x[valid]**1.61
        b[valid] = -0.527 * x[valid]**1.61

        # optical, using O'Donnell (1994) values
        valid = (1.1 <= x) & (x < 3.3)
        y = x[valid] - 1.82
        coef_a = np.array([-0.505, 1.647, -0.827, -1.718, 1.137, 0.701, -0.609,
            0.104, 1.])
        coef_b = np.array([3.347, -10.805, 5.491, 11.102, -7.985, -3.989, 2.908,
            1.952, 0.])
        a[valid] = np.polyval(coef_a,y)
        b[valid] = np.polyval(coef_b,y)

        # UV
        valid = (3.3 <= x) & (x < 8)
        y = x[valid]
        f_a = np.zeros(y.size)
        f_b = np.zeros(y.size)
        select = (y >= 5.9)
        yselect = y[select] - 5.9

        f_a[select] = -0.04473 * yselect**2 - 0.009779 * yselect**3
        f_b[select] = 0.2130 * yselect**2 + 0.1207 * yselect**3
        a[valid] = 1.752 - 0.316*y - (0.104 / ((y-4.67)**2 + 0.341)) + f_a
        b[valid] = -3.090 + 1.825*y + (1.206 / ((y-4.62)**2 + 0.263)) + f_b

        # far-UV CCM89 extrapolation
        valid = (8 <= x) & (x < 11)
        y = x[valid] - 8.
        coef_a = np.array([-0.070, 0.137, -0.628, -1.073])
        coef_b = np.array([0.374, -0.420, 4.257, 13.670])
        a[valid] = np.polyval(coef_a,y)
        b[valid] = np.polyval(coef_b,y)

        # Overwrite UV with GCC09 model if applicable. Not an extrapolation.
        if model == 'gcc09':
            valid = (3.3 <= x) & (x < 11)
            y = x[valid]
            f_a = np.zeros(y.size)
            f_b = np.zeros(y.size)
            select = (5.9 <= y)
            yselect = y[select] - 5.9
            f_a[select] = -0.110 * yselect**2 - 0.0099 * yselect**3
            f_b[select] = 0.537 * yselect**2 + 0.0530 * yselect**3
            a[valid] = 1.896 - 0.372*y - (0.0108 / ((y-4.57)**2 + 0.0422)) + f_a
            b[valid] = -3.503 + 2.057*y + (0.718 / ((y-4.59)**2 + 0.0530*3.1)) + f_b

        if isinstance(ebv, np.ndarray):
            if (len(ebv.shape) > 1):
                a = a[np.newaxis, np.newaxis, :]; b = b[np.newaxis, np.newaxis, :]
                ebv = ebv[:, :, np.newaxis]
        
        a_v = ebv * r_v
        a_lambda = a_v * (a + b/r_v)
        reddening_curve = 10**(0.4 * a_lambda)

        return reddening_curve

    def _calculate_mean_extinction(self, wavelength, response):
        """
        Compute the mean extinction over a bandpass.

        This calculates the response-weighted mean of the extinction A(λ)
        over the given bandpass.

        Parameters
        ----------
        wavelength : array_like
            Wavelengths in Angstroms.
        response : array_like
            Filter transmission values corresponding to wavelength.

        Returns
        -------
        A_mean : float
            The mean extinction in magnitudes for the given bandpass.
        """
        # Ensure the inputs are numpy arrays
        wavelength = np.asarray(wavelength)
        response = np.asarray(response)
        
        # Get reddening curve (extinction in magnitudes for each wavelength)
        red_curve_multiplier = self.reddening_ccm(wavelength, ebv=self.ebv, a_v=None, r_v=3.1, model='ccm89')
        A_lambda = 2.5 * np.log10(red_curve_multiplier)

        # Calculate response-weighted mean extinction:
        # A_mean = integral(T(λ) * A(λ) dλ) / integral(T(λ) dλ)
        numerator = np.trapezoid(response * A_lambda, x=wavelength)
        denominator = np.trapezoid(response, x=wavelength)
        
        if denominator == 0:
            return 0.0

        A_mean = numerator / denominator
        return A_mean
