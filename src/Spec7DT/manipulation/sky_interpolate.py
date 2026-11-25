import numpy as np

def interpolate_sky(image_data, galaxy_name, observatory, band, image_set):
    # gauss_kernal = Gaussian2DKernel(x_stddev=1)
    # interp_img = convolve(image_data, gauss_kernal, normalize_kernel=True, nan_treatment='interpolate')
    invalid = ~np.isfinite(image_data)
    interp_img = np.nan_to_num(image_data, nan=0.0)
    
    image_set.update_data(interp_img, galaxy_name, observatory, band)
    # Zero-filled pixels carry no measurement: keep their error non-finite so
    # unit conversion can mark them with the invalid-error fill value.
    error = image_set.error.get(galaxy_name, {}).get(observatory, {}).get(band)
    if error is not None and np.any(invalid):
        error = np.array(error, dtype=np.float32, copy=True)
        error[invalid] = np.nan
        image_set.update_error(error, galaxy_name, observatory, band)