import math
import numpy as np
from astroquery.ipac.ned import Ned
from astropy.wcs import WCS
from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.psf import PSFPhotometry, MoffatPSF
from photutils.aperture import CircularAperture

from ..utils.utility import useful_functions

class Masking:
    def __init__(self):
        pass
    
    @classmethod
    def adapt_mask(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set, manual):
        if ("max" not in image_set.psf[galaxy_name]) or (image_set.psf[galaxy_name]["max"] == -1):
            psf_list = useful_functions.extract_values_recursive(image_set.psf, galaxy_name)
            sig_t = np.max(psf_list)
            sig_3 = np.median(psf_list) + 3 * np.std(psf_list)
            
            sig_t = np.min([sig_t, sig_3])
            
            image_set.psf = (galaxy_name, "max", sig_t)
            
        fwhm = image_set.psf[galaxy_name]["max"]  # in "
        pixel_scale = np.abs(header.get("CD1_1", 1.1e-4)) * 3600
        fwhm = fwhm / pixel_scale  # in pixel
        
        mask_image, masked_image, _ = cls.make_mask(cls, image_data, header, galaxy_name, fwhm, manual)
        masked_err = np.where(mask_image, 999.0, error_data)
        
        image_set.update_data(masked_image, galaxy_name, observatory, band)
        image_set.update_error(masked_err, galaxy_name, observatory, band)
    
    
    def make_mask(self, image, header, galaxy, psf_fwhm, manual):
        mask, masked, sky = self.py2dmask(self, image, header, galaxy, psf_fwhm)
        
        if manual is not None:
            star_mask = self.manual_mask(self, image, header, psf_fwhm, manual)
            masked *= star_mask
            
        return mask, masked, sky
    
    
    def manual_mask(self, image, header, psf_fwhm, manual):
        coords = manual.get("coord", None)
        rads = manual.get("radius", None)
        
        if any(i is None for i in [coords, rads]):
            raise ValueError("Manual values are not input.")
        elif len(coords) != len(rads):
            raise ValueError("Length of coordinates and radii are not same.")
        
        wcs = WCS(header)
        mask = np.zeros_like(image)
        
        for coord, rad in zip(coords, rads):
            star_loc = wcs.all_world2pix(coord[0], coord[1], 0)
            star_mask = CircularAperture(positions=star_loc, r=rad).to_mask().to_image(shape=image.shape)
            if star_mask is None:
                continue
            star_mask = np.where(star_mask > 0, 0, 1)
            mask += star_mask
            
        mask = np.where(mask/np.max(mask) == 1, 1, 0)
        
        return mask
    
    
    def py2dmask(self, image, header, galaxy, psf_fwhm):
        ra, dec = Ned.query_object(galaxy)['RA', 'DEC'][0]
        
        wcs = WCS(header)
        x, y = wcs.all_world2pix(ra, dec, 0)
        
        from astropy.stats import sigma_clipped_stats
        mean, median, std = sigma_clipped_stats(image, sigma=10.0, mask=np.where(image == 0, True, False))
        
        from photutils.segmentation import detect_sources
        segment_map = detect_sources(image, threshold=5 * std, npixels=30)
        try:
            segment_map.remove_label(label=int(segment_map.data[int(y), int(x)]), relabel=True)
        except Exception as e:
            print(e)
            print("Un-removed")
            pass
        
        checksum = np.sum(np.where(segment_map.data != 0, 0, 1))
        if checksum != 0:
            masked_image = np.where(segment_map.data == 0, image, np.nan)
        else:
            masked_image = image
        
        return np.where(segment_map.data == 0, False, True), masked_image, np.full_like(masked_image, fill_value=median)
    

    def own_mask(self, image, header, galaxy, psf_fwhm):
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        
        daofind = DAOStarFinder(fwhm=psf_fwhm, threshold=15.*std)
        sources = daofind(image - median)
        
        if sources is None:
            print('No sources found.')
            return image, image, image

        self.x0, self.y0, self.a, self.b, self.theta = useful_functions.get_galaxy_radius(image)

        dist = self.is_point_in_rotated_ellipse(self, sources['xcentroid'], sources['ycentroid'])

        good = dist > 1
        filtered_sources = sources[good]
        if not filtered_sources.indices:
            return image, image, image

        psf_model = MoffatPSF()
        psf_model.alpha.fixed = False
        psf_model.flux.fixed = False
        fit_shape =(int(psf_fwhm*2.0) * 2 + 1, int(psf_fwhm*2.0) * 2 + 1)
        bkgstat = MMMBackground()
        localbkg_estimator = LocalBackground(5, 10, bkgstat)
        
        psfphot = PSFPhotometry(psf_model, fit_shape)
        phot = psfphot(image - median, init_params=filtered_sources["xcentroid", "ycentroid", "flux"],
                       localbkg_estimator=localbkg_estimator)
        
        if phot is None:
            print('No PSF photometry found.')
            return image, image, image
        
        print(f"Phot: {phot}")
        
        resid = psfphot.make_residual_image(image - median)
        mask_image = np.where(image - median - resid > 0, 1, 0)
        masked_image = resid.copy()
        
        return mask_image, masked_image, np.full_like(masked_image, fill_value=median)


    def __make_mask(self, image, header, galaxy, psf_fwhm):
        psf_fwhm = np.abs(psf_fwhm)
        ra, dec = Ned.query_object(galaxy)['RA', 'DEC'][0]
        image_x, image_y = image.shape
        box_shape = [int(image_x/5), int(image_y/5)]
        filter_shape = [int(psf_fwhm*2.0) * 2 + 1, int(psf_fwhm*2.0) * 2 + 1]
        
        wcs = WCS(header)
        x, y = wcs.all_world2pix(ra, dec, 0)

        bkg_estimator = MedianBackground()
        try:    
            bkg = Background2D(image, box_shape, filter_size=filter_shape, bkg_estimator=bkg_estimator)
        except ValueError:
            print('ValueError occured. Try Smaller Background size.')
            bkg = Background2D(image, (int(box_shape[0]/5), int(box_shape[1]/5)), filter_size=filter_shape, bkg_estimator=bkg_estimator)
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
        daofind = DAOStarFinder(fwhm=psf_fwhm, threshold=15.*std)
        sources = daofind(masked_image - mean)
        
        if sources is None:
            print('No sources found.')
            return masked_image, masked_image, masked_image

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
