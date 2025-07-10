import numpy as np

class CutRegion:
    
    @classmethod
    def cutout_region(cls, box_size, image_data, error_data, galaxy_name, observatory, band, image_set):
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
        ny, nx = img.shape
        cx, cy = nx // 2, ny // 2  # center coordinates

        if _shape == 'box':
            hs = int(size // 2)
            cut = np.zeros_like(img)
            cut[cy-hs:cy+hs, cx-hs:cx+hs] = img[cy-hs:cy+hs, cx-hs:cx+hs]
            return cut

        elif _shape == 'circle':
            r = size / 2
            y, x = np.ogrid[:ny, :nx]
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
            return img * mask.astype(img.dtype)

        elif _shape == 'ellipse':
            # Determine semiaxes in pixels
            if isinstance(size, (list, tuple, np.ndarray)):
                w, h = size
            else:
                w = h = size
            a = w / 2  # semimajor axis
            b = h / 2  # semiminor axis

            # Create an ellipse mask
            y, x = np.ogrid[:ny, :nx]
            # Standard ellipse equation (centered at cx,cy)
            mask = ((x - cx)**2 / a**2 + (y - cy)**2 / b**2) <= 1.0
            
            return img * mask.astype(img.dtype), error * mask.astype(error.dtype)

        else:
            raise ValueError(f"Unknown shape '{_shape}'. Choose 'box', 'circle', or 'ellipse'.")
