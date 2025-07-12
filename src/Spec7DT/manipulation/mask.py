import math
import numpy as np
from astroquery.ipac.ned import Ned
from astropy.wcs import WCS
from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.psf import PSFPhotometry, MoffatPSF

from ..utils.utility import useful_functions

class Masking:
    def __init__(self):
        pass
    
    @classmethod
    def adapt_mask(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set):
        fwhm = image_set.psf[galaxy_name][observatory][band]  # in "
        pixel_scale = np.abs(header.get("CD1_1", 1.1e-4)) * 3600
        fwhm = fwhm / pixel_scale  # in pixel
        
        mask_image, masked_image, _ = cls.make_mask(cls, image_data, header, galaxy_name, fwhm)
        masked_err = np.where(mask_image, 999.0, error_data)
        
        image_set.update_data(masked_image, galaxy_name, observatory, band)
        image_set.update_error(masked_err, galaxy_name, observatory, band)
        

    def make_mask(self, image, header, galaxy, psf_fwhm):
        ra, dec = Ned.query_object(galaxy)['RA', 'DEC'][0]

        wcs = WCS(header)
        x, y = wcs.all_world2pix(ra, dec, 0)

        bkg_estimator = MedianBackground()
        try:    
            bkg = Background2D(image, (500, 500), filter_size=(13, 13), bkg_estimator=bkg_estimator)
        except ValueError:
            print('ValueError occured. Try Smaller Background size.')
            bkg = Background2D(image, (100, 100), filter_size=(13, 13), bkg_estimator=bkg_estimator)
        threshold = 1.5*bkg.background_rms

        segment_map = detect_sources(image, threshold, npixels=5)
        if segment_map == None:
            return image, image, image

        sky_map = np.nonzero(segment_map.data)
        sky_image = image.copy()
        sky_image[sky_map] = np.nan

        label_main = segment_map.data[int(y), int(x)]
        if label_main != 0:
            segment_map.remove_labels([label_main])

        mask = np.nonzero(segment_map.data)
        masked_image = image.copy()
        masked_image[mask] = np.nan
            
        mask_image = np.zeros_like(image)
        mask_image[mask] = image[mask]
        
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        daofind = DAOStarFinder(fwhm=psf_fwhm, threshold=30.*std)
        sources = daofind(masked_image - mean)

        self.x0, self.y0, self.a, self.b, self.theta = useful_functions.get_galaxy_radius(image)

        dist = self.is_point_in_rotated_ellipse(self, sources['xcentroid'], sources['ycentroid'])

        good = dist > 1
        filtered_sources = sources[good]
        if not filtered_sources.indices:
            return image, image, image

        psf_model = MoffatPSF()
        psf_model.alpha.fixed = False
        psf_model.flux.fixed = False
        fit_shape = (31, 31) 
        psfphot = PSFPhotometry(psf_model, fit_shape,
                                aperture_radius=psf_fwhm*5)
        phot = psfphot(masked_image - mean, init_params=filtered_sources["xcentroid", "ycentroid", "flux"])
        if phot is None:
            print('No PSF photometry found.')
            return masked_image, masked_image, masked_image
        
        resid = psfphot.make_residual_image(masked_image - mean)
        mask_image = np.where(masked_image - mean - resid > 0, 1, 0)
        masked_image = resid.copy()
        
        return mask_image, masked_image, sky_image
        

    def is_point_in_rotated_ellipse(self, x, y):

        dx = x - self.x0
        dy = y - self.y0

        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        x_rot =  dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        value = (x_rot / self.a)**2 + (y_rot / self.b)**2
        return value
