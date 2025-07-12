from astropy.convolution import convolve, Gaussian2DKernel
import numpy as np

def interpolate_sky(image_data, galaxy_name, observatory, band, image_set):
    # gauss_kernal = Gaussian2DKernel(x_stddev=1)
    # interp_img = convolve(image_data, gauss_kernal, normalize_kernel=True, nan_treatment='interpolate')
    interp_img = np.where(image_data == np.nan, 0, image_data)
    
    image_set.update_data(interp_img, galaxy_name, observatory, band)