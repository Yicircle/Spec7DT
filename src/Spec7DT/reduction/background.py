import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground, SExtractorBackground

def backgroundSubtraction(image_set, image_data, galaxy_name, observatory, band):
    im_x, im_y = image_data.shape
    box_size = (int(im_x * 0.25), int(im_y * 0.25))
    filter_size = (int(im_x * 5e-3) * 2 + 1, int(im_y * 5e-3) * 2 + 1)
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    coverage_mask = (image_data == 0) | np.isnan(image_data)
    bkg = Background2D(image_data, box_size=box_size, filter_size=filter_size, coverage_mask=coverage_mask, fill_value=0.0,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, exclude_percentile=50.0)
    bkg_map = bkg.background
    
    # sigma_clip = SigmaClip(sigma=3.0)
    # bkg = SExtractorBackground(sigma_clip)
    # bkg_value = bkg.calc_background(image_data)
    
    image_set.update_data(image_data - bkg_map, galaxy_name, observatory, band)
    