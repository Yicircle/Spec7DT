class Bin:
    def __init__(self):
        pass
    
    @classmethod
    def do_binning(cls, bin_size, image_data, error_data, galaxy_name, observatory, band, image_set):
        # if bin_size is None:
        #     bin_size = image_data.shape[0]
        binned_img = cls.binning(image_data, bin_size, bin_size)
        binned_err = cls.binning_err(error_data, bin_size, bin_size)
        image_set.update_data(binned_img, galaxy_name, observatory, band)
        image_set.update_error(binned_err, galaxy_name, observatory, band)
        
    def binning(image, bin_x, bin_y):
        return image.reshape(bin_x, image.shape[0] // bin_x, bin_y, image.shape[1] // bin_y).sum(3).sum(1)

    def binning_err(image, bin_x, bin_y):
        image = image ** 2
        image = image.reshape(bin_x, image.shape[0] // bin_x, bin_y, image.shape[1] // bin_y).sum(3).sum(1)
        return image ** (0.5)