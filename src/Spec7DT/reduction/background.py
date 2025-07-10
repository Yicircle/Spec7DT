from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

def backgroundSubtraction(image_set, image_data, galaxy_name, observatory, band):
    im_x, im_y = image_data.shape
    box_size = (int(im_x * 0.1), int(im_y * 0.1))
    filter_size = (int(im_x * 5e-3) * 2 + 1, int(im_y * 5e-3) * 2 + 1)
    
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image_data, box_size=box_size, filter_size=filter_size,
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bkg_map = bkg.background
    
    image_set.update_data(image_data - bkg_map, galaxy_name, observatory, band)
    