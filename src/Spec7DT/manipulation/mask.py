import numpy as np
from astroquery.ipac.ned import Ned
from astropy.wcs import WCS
from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.psf import PSFPhotometry, MoffatPSF

class Masking:
    def __init__(self):
        pass
    
    @classmethod
    def adapt_mask(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set):
        mask_image, masked_image, _ = cls.make_mask(cls, image_data, header, galaxy_name)
        masked_err = np.where(mask_image, 999.0, error_data)
        
        image_set.update_data(masked_image, galaxy_name, observatory, band)
        image_set.update_error(masked_err, galaxy_name, observatory, band)
        

    def make_mask(self, image, header, galaxy):
        ra, dec = Ned.query_object(galaxy)['RA', 'DEC'][0]

        wcs = WCS(header)
        x, y = wcs.all_world2pix(ra, dec, 0)

        bkg_estimator = MedianBackground()
        try:    
            bkg = Background2D(image, (500, 500), filter_size=(13, 13), bkg_estimator=bkg_estimator)
        except ValueError:
            print('ValueError occured. Try Smaller Background size.')
            bkg = Background2D(image, (200, 200), filter_size=(13, 13), bkg_estimator=bkg_estimator)
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
        daofind = DAOStarFinder(fwhm=6.0, threshold=100.*std)
        sources = daofind(masked_image - mean)  # Table with 'xcentroid', 'ycentroid', etc.

        # 2. Compute center and a small exclusion radius (in pixels)
        excl_radius = 200.0  # e.g. 5 pixels

        # 3. Measure distance of each source to center
        dx = sources['xcentroid'] - x
        dy = sources['ycentroid'] - y
        dist = np.hypot(dx, dy)

        # 4. Filter out the central source(s)
        good = dist > excl_radius
        filtered_sources = sources[good]

        psf_model = MoffatPSF(alpha=5.0)
        psf_model.alpha.fixed = False
        psf_model.flux.fixed = False
        fit_shape = (31, 31) 
        psfphot = PSFPhotometry(psf_model, fit_shape,
                                aperture_radius=12.0)
        phot = psfphot(masked_image - mean, init_params=filtered_sources["xcentroid", "ycentroid", "flux"])
        if phot is None:
            print('No PSF photometry found.')
            return masked_image, masked_image, masked_image
        
        resid = psfphot.make_residual_image(masked_image - mean)
        mask_image = np.where(masked_image - mean - resid > 0, 0, 1)
        masked_image = resid.copy()
        
        return mask_image, masked_image, sky_image