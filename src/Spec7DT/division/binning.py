from reproject.mosaicking import find_optimal_celestial_wcs
from astropy import units as u
from astropy.wcs import WCS
from ..utils.utility import useful_functions
from astropy.nddata import NDData

class Bin:
    def __init__(self):
        pass
    
    @classmethod
    def do_binning(cls, bin_size, image_data, error_data, galaxy_name, observatory, band, image_set):
        im_header = image_set.header[galaxy_name][observatory][band]
        pix_scale_ori = useful_functions.get_pixel_scale(im_header)
        
        total_size = int(image_data.shape[0] // bin_size)
        binned_img = cls.binning(image_data, total_size, total_size)
        binned_err = cls.binning_err(error_data, total_size, total_size)
        wcs_header = WCS(im_header)
        wcs_out, _ = find_optimal_celestial_wcs(NDData(image_data, wcs=WCS(im_header)), resolution=pix_scale_ori * bin_size * u.arcsec)
        
        image_set.update_data(binned_img, galaxy_name, observatory, band)
        image_set.update_error(binned_err, galaxy_name, observatory, band)
        image_set.header[galaxy_name][observatory][band] = wcs_out.to_header()
        
    def binning(image, bin_x, bin_y):
        return image.reshape(bin_x, image.shape[0] // bin_x, bin_y, image.shape[1] // bin_y).sum(3).sum(1)

    def binning_err(image, bin_x, bin_y):
        image = image ** 2
        image = image.reshape(bin_x, image.shape[0] // bin_x, bin_y, image.shape[1] // bin_y).sum(3).sum(1)
        return image ** (0.5)