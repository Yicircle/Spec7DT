import numpy as np

from ..utils.utility import useful_functions

class CutRegion:
    
    @classmethod
    def cutout_region(cls, box_size, image_data, error_data, galaxy_name, observatory, band, image_set):
        if box_size == None or not box_size:
            x, y, a, b, th = useful_functions.get_galaxy_radius(image_data)
            a = 1.5 * a; b = 1.5 * b
            box_size = (x, y, a, b, th)
        
        cut_img, cut_error = cls.get_cutout(image_data, error_data, box_size, 'ellipse')
        image_set.update_data(cut_img, galaxy_name, observatory, band)
        image_set.update_error(cut_error, galaxy_name, observatory, band)

    def get_cutout(img, error, size, _shape: str='box'):
        """
        Mask out everything except a central region of shape 'box', 'circle', or 'ellipse'.

        Parameters
        ----------
        img : 2D ndarray
            Input image.
        size : float or tuple of floats
            - If _shape in {'box','circle'}: scalar = side‐length (box) or diameter (circle).
            - If _shape=='ellipse': either
                * scalar = major and minor axes
                * tuple (width, height) in pixels for major/minor axes.
        _shape : {'box','circle','ellipse'}
            Shape of the kept region.

        Returns
        -------
        cutout : 2D ndarray
            Same shape as `img`, with pixels **outside** the requested region set to zero.
        """
        x0, y0, a, b, theta = size
        

        if _shape == 'box':
            from photutils.aperture import RectangularAperture
            mask = RectangularAperture((x0, y0), w=a, h=b, theta=theta)
            mask_im = mask.to_mask().to_image(shape=img.shape)
            
            return img * mask_im.astype(img.dtype), error * mask_im.astype(error.dtype)

        elif _shape == 'circle':
            from photutils.aperture import CircularAperture
            mask = CircularAperture((x0, y0), r=a)
            mask_im = mask.to_mask().to_image(shape=img.shape)
            
            return img * mask_im.astype(img.dtype), error * mask_im.astype(error.dtype)

        elif _shape == 'ellipse':
            from photutils.aperture import EllipticalAperture
            mask = EllipticalAperture((x0, y0), a=a, b=b, theta=theta)
            mask_im = mask.to_mask().to_image(shape=img.shape)
            
            return img * mask_im.astype(img.dtype), error * mask_im.astype(error.dtype)

        else:
            raise ValueError(f"Unknown shape '{_shape}'. Choose 'box', 'circle', or 'ellipse'.")
