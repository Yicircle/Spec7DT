import numpy as np
from scipy.stats import mode

from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.convolution import convolve_fft, Gaussian2DKernel

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

class PointSpreadFunction:
    
    def __call__(self):
        pass
    
    @classmethod
    def extract(cls, image_data, header, galaxy_name, observatory, band, image_set):    
        fwhm_val = cls.measure_psf_fwhm(cls, image_data, header, threshold_sigma=15)
        
        image_set.set_psf(galaxy_name, observatory, band, fwhm_val) # in " fwhm * pixel_scale
    
    @classmethod
    def convolution(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set):
        """
        Convolve `image` with a Gaussian kernel of width `sigma_extra_pix` (pixels).
        If sigma_extra_pix==0, return original image.
        """
        pixel_scale = np.abs(header.get("CD1_1", 1.1e-4)) * 3600  # in "
        
        psf_list = image_set.get_psf(galaxy_name)
        sig_i = image_set.psf[galaxy_name][observatory][band]
        sig_t = np.max(psf_list)
        sigma_extra = np.sqrt(sig_t**2 - sig_i**2) / pixel_scale
        
        if sigma_extra <= 0:
            return image_data.copy()
        kernel = Gaussian2DKernel(x_stddev=sigma_extra)
        convolved_img = convolve_fft(
            image_data, kernel,
            normalize_kernel=True,
            nan_treatment='interpolate'
        )
        convolved_err = convolve_fft(
            error_data, kernel,
            normalize_kernel=True,
            nan_treatment='interpolate'
        )
        image_set.update_data(convolved_img, galaxy_name, observatory, band)
        image_set.update_error(convolved_err, galaxy_name, observatory, band)
    

    def measure_fwhm_gaussian(image, x_center, y_center, box_size=21):
        """
        Measure FWHM by fitting 2D Gaussian to PSF
        """
        # Extract cutout around star
        y, x = np.ogrid[:box_size, :box_size]
        x_start = int(x_center - box_size//2)
        y_start = int(y_center - box_size//2)
        
        cutout = image[y_start:y_start+box_size, x_start:x_start+box_size]
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[:box_size, :box_size]
        
        # Initial parameter guess
        amplitude = np.max(cutout)
        x_mean = box_size // 2
        y_mean = box_size // 2
        
        # Fit 2D Gaussian
        g_init = models.Gaussian2D(amplitude=amplitude, 
                                x_mean=x_mean, y_mean=y_mean,
                                x_stddev=box_size * 0.05, y_stddev=box_size * 0.05)
        fit_g = fitting.TRFLSQFitter()
        g = fit_g(g_init, x_grid, y_grid, cutout)
        
        # Convert stddev to FWHM
        fwhm_x = 2.355 * g.x_stddev.value
        fwhm_y = 2.355 * g.y_stddev.value
        fwhm_avg = (fwhm_x + fwhm_y) / 2
        
        return fwhm_avg

    def measure_psf_fwhm(self, image, header, threshold_sigma=15):
        """
        Complete pipeline: detect stars and measure FWHM
        """
        im_x, im_y = image.shape
        box_size = int((im_x + im_y) * 0.1 / 2)
        mean, median, std = sigma_clipped_stats(image, sigma=10.0)
        
        # Star detection
        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma*std)
        sources = daofind(image)
        
        if sources is None:
            return -1.0
        
        # Measure FWHM for each detected star
        fwhm_measurements = []
        
        for source in sources:
            x, y = source['xcentroid'], source['ycentroid']
            
            margin = 0.1
            # Skip stars too close to edges
            if (x > im_x * margin and x < im_x * (1 - margin) and 
                y > im_y * margin and y < im_y * (1 - margin)):
                
                try:
                    fwhm = self.measure_fwhm_gaussian(image, x, y, box_size=box_size)
                    if not np.isnan(fwhm) and fwhm > 0:
                        fwhm_measurements.append(fwhm)
                except:
                    continue
        
        fwhm, _ = mode(fwhm_measurements, nan_policy='omit')
        pixel_scale = np.abs(header.get("CD1_1", 1.1e-4)) * 3600  # in "
        
        return fwhm * pixel_scale
