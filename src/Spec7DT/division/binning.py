from astropy.wcs import WCS
from ..utils.utility import useful_functions

class Bin:
    def __init__(self):
        pass
    
    @classmethod
    def do_binning(cls, bin_size, image_data, error_data, galaxy_name, observatory, band, image_set):
        im_header = image_set.header[galaxy_name][observatory][band]
        if bin_size < 1:
            raise ValueError("bin_size must be >= 1")

        binned_y = int(image_data.shape[0] // bin_size)
        binned_x = int(image_data.shape[1] // bin_size)
        if binned_y < 1 or binned_x < 1:
            raise ValueError("bin_size is larger than the image dimensions")

        crop_y = binned_y * bin_size
        crop_x = binned_x * bin_size
        image_data = image_data[:crop_y, :crop_x]
        error_data = error_data[:crop_y, :crop_x]

        binned_img = cls.binning(image_data, binned_y, binned_x)
        binned_err = cls.binning_err(error_data, binned_y, binned_x)
        wcs_out = cls.binned_wcs(WCS(im_header), bin_size)
        
        header_clean = im_header.copy()
        wcs_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
                    'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2',
                    'CUNIT1', 'CUNIT2', 'RADESYS', 'LONPOLE', 'LATPOLE']
        
        for key in wcs_keys:
            if key in header_clean:
                del header_clean[key]
        
        for key in list(header_clean.keys()):
            if key.startswith('PV'):
                del header_clean[key]
        
        updated_header = useful_functions.update_header(header_clean, wcs_out.to_header())
        updated_header["NAXIS1"] = binned_x
        updated_header["NAXIS2"] = binned_y
        
        image_set.update_data(binned_img, galaxy_name, observatory, band)
        image_set.update_error(binned_err, galaxy_name, observatory, band)
        image_set.update_header(updated_header, galaxy_name, observatory, band)

    @staticmethod
    def binned_wcs(wcs, bin_size):
        wcs_out = wcs.deepcopy()
        wcs_out.wcs.crpix = (wcs_out.wcs.crpix - 0.5) / bin_size + 0.5

        try:
            if wcs_out.wcs.has_cd():
                wcs_out.wcs.cd = wcs_out.wcs.cd * bin_size
            else:
                wcs_out.wcs.cdelt = wcs_out.wcs.cdelt * bin_size
        except AttributeError:
            wcs_out.wcs.cdelt = wcs_out.wcs.cdelt * bin_size

        return wcs_out
        
    def binning(image, bin_x, bin_y):
        return image.reshape(bin_x, image.shape[0] // bin_x, bin_y, image.shape[1] // bin_y).sum(3).sum(1)

    def binning_err(image, bin_x, bin_y):
        image = image ** 2
        image = image.reshape(bin_x, image.shape[0] // bin_x, bin_y, image.shape[1] // bin_y).sum(3).sum(1)
        return image ** (0.5)
