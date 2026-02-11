import numpy as np
import math
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog


class Observatories:
    """Class to handle different observatories and their properties."""
    def __init__(self):
        self.optical_obs = self._opticals()
        self.ir_obs = self._infrareds()
        self.uv_obs = self._ultraviolet()
        self.radio_obs = self._radio()
        self.observatories = list(set(self.optical_obs + self.ir_obs + self.uv_obs + self.radio_obs))
        
    def _opticals(self):
        """Return a list of optical observatories."""
        return ['SDSS', 'PS1', 'DECam', 'DES', 'LSST', 'Pan-STARRS', 'Subaru', '7DT', 'SkyMapper', 'SLOAN']
    
    def _infrareds(self):
        """Return a list of infrared observatories."""
        return ['WISE', 'Spitzer', 'IRAC', 'MIPS', 'Herschel', 'JWST', 'VISTA', 'UKIDSS', '2MASS', 'SPHEREx', 'PACS', 'SPIRE']
    
    def _ultraviolet(self):
        """Return a list of ultraviolet observatories."""
        return ['GALEX']
    
    def _radio(self):
        """Return a list of radio observatories."""
        return ['VLA', 'ALMA', 'SKA', 'MeerKAT', 'GMRT']
    
    @classmethod
    def get_observatories(cls):
        """Return a list of all observatories."""
        return cls().observatories
    

class useful_functions:
    @classmethod
    def get_redshift(cls, galaxy_name):
        """
        Query redshift from NED using galaxy name.
        
        Parameters:
        -----------
        galaxy_name : str
            Name of the galaxy (e.g., 'NGC 3627', 'M81', 'NGC4321')
        
        Returns:
        --------
        float or None
            Redshift value, or None if not found
        """
        from astroquery.ned import Ned
        
        try:
            # Query basic information from NED
            result_table = Ned.query_object(galaxy_name)
            
            if len(result_table) > 0:
                # Get the redshift value
                redshift = result_table['Redshift'][0]
                
                # Check if redshift is valid (not masked or NaN)
                if not np.ma.is_masked(redshift) and not np.isnan(redshift):
                    return float(redshift)
                else:
                    print(f"No redshift data available for {galaxy_name}")
                    return None
            else:
                print(f"Galaxy {galaxy_name} not found in NED")
                return None
                
        except Exception as e:
            print(f"Error querying {galaxy_name}: {str(e)}")
            return None
    
    @classmethod
    def get_galaxy_radius(cls, image):
        threshold = detect_threshold(image, nsigma=3.0)

        segm = detect_sources(image, threshold, npixels=5)
        if segm is None:
            print("No sources detected.")
            a, b = image.shape
            x0, y0 = image.shape[0]/2, image.shape[1]/2
            theta = 0
            return x0, y0, a, b, theta

        catalog = SourceCatalog(image, segm)
        gal = max(catalog, key=lambda src: src.area)

        x0, y0 = gal.xcentroid, gal.ycentroid
        a, b = gal.semimajor_sigma.value*2, gal.semiminor_sigma.value*2
        theta = math.radians(gal.orientation.value)
        return x0, y0, a, b, theta
    
    @staticmethod
    def get_sky_loc(location):
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.ned import Ned
        
        if isinstance(location, str):
            ra, dec = Ned.query_object(location)['RA', 'DEC'][0]
            
        sky_loc = SkyCoord(ra=ra, dec=dec, unit="deg")
        return sky_loc
    
    @staticmethod
    def find_rec(N):
        # Start from the square root of N and work downwards
        num_found = False
        while not num_found:
            for k in range(int(N ** 0.5), 0, -1):
                if N % k == 0:  # k must divide N
                    l = N // k  # Calculate l
                    # Check the condition that neither exceeds twice the other
                    if k <= 2 * l and l <= 2 * k:
                        num_found = True
                        return k, l
            N = N + 1
        return None, None  # Return None if no valid pair is found
    
    @staticmethod
    def get_pixel_scale(header, typical=1.0):
        cd1_1 = np.abs(header.get("CD1_1", header.get("PC1_1", header.get("CDELT1", typical))))
        cd1_2 = np.abs(header.get("CD1_2", header.get("PC1_2", 0.00)))
        pixel_scale = 3600 * np.sqrt(cd1_1 ** 2 + cd1_2 ** 2)
        return pixel_scale
    
    @staticmethod
    def extract_values_recursive(dictionary, key):
        """
        Alternative recursive approach that handles arbitrary nesting depth.
        
        Args:
            dictionary: Dictionary with nested structure
            key: The key at level1 to extract values from
        
        Returns:
            List of all values found in the nested structure
        """
        def _extract_all_values(obj):
            """Recursively extract all values from nested dict/list structures."""
            if isinstance(obj, dict):
                values = []
                for v in obj.values():
                    values.extend(_extract_all_values(v))
                return values
            elif isinstance(obj, list):
                values = []
                for item in obj:
                    values.extend(_extract_all_values(item))
                return values
            else:
                return [obj]
        
        if key not in dictionary:
            return []
        
        return _extract_all_values(dictionary[key])
    
    @classmethod
    def tour_nested_dict_with_keys(cls, dictionary):
        """
        Tour through a 3-level nested dictionary and yield keys and values in order.
        
        Args:
            dictionary: Dictionary with structure dict[level1][level2][level3]
        
        Yields:
            tuple: (keys_tuple, value) where keys_tuple contains (level1_key, level2_key, level3_key)
        """
        for level1_key, level1_dict in dictionary.items():
            for level2_key, level2_dict in level1_dict.items():
                for level3_key, value in level2_dict.items():
                    yield (level1_key, level2_key, level3_key), value


    def get_all_keys_and_values(self, my_dict):
        """
        Get all keys and values from a 3-level nested dictionary as a list.
        
        Args:
            my_dict: Dictionary with structure dict[level1][level2][level3]
        
        Returns:
            list: List of tuples [(keys_tuple, value), ...] where keys_tuple contains (level1_key, level2_key, level3_key)
        """
        result = []
        for level1_key, level1_dict in my_dict.items():
            for level2_key, level2_dict in level1_dict.items():
                for level3_key, value in level2_dict.items():
                    result.append(((level1_key, level2_key, level3_key), value))
        return result


    def tour_nested_dict_recursive(self, obj, current_keys=()):
        """
        Recursive function to tour through arbitrarily nested dictionaries.
        
        Args:
            obj: Dictionary or value to traverse
            current_keys: Current key path (used internally)
        
        Yields:
            tuple: (keys_tuple, value) where keys_tuple contains all keys in the path
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                yield from self.tour_nested_dict_recursive(value, current_keys + (key,))
        else:
            yield current_keys, obj
            
    @classmethod
    def pa_sky_to_image(cls, x, y, pa_east_of_north, wcs, offset_deg=0.001):
        """
        Convert Position Angle from astronomical standard (East of North) 
        to image coordinates (Right of Top) for plotting.
        
        Parameters
        ----------
        x : float
            X pixel coordinate (column)
        y : float
            Y pixel coordinate (row)
        pa_east_of_north : float
            Position angle in degrees, measured East of North
            (astronomical standard: 0° = North, 90° = East)
        wcs : astropy.wcs.WCS
            WCS object for coordinate transformation
        offset_deg : float, optional
            Small offset in degrees for numerical calculation (default: 0.001)
        
        Returns
        -------
        pa_image : float
            Position angle in image coordinates (degrees)
            Measured Right of Top (0° = up, 90° = right)
            This is the angle for matplotlib plotting
        """
        # Get sky coordinate at the reference point
        sky_center = wcs.pixel_to_world(x, y)
        
        # Calculate offset point in sky coordinates along the PA direction
        pa_rad = np.radians(pa_east_of_north)
        
        # East of North: 0° = North, 90° = East
        # dRA increases toward East, dDec increases toward North
        dra = offset_deg * np.sin(pa_rad) / np.cos(np.radians(sky_center.dec.deg))
        ddec = offset_deg * np.cos(pa_rad)
        
        # Create offset sky coordinate
        sky_offset = SkyCoord(
            ra=sky_center.ra + dra * u.deg,
            dec=sky_center.dec + ddec * u.deg,
            frame=sky_center.frame
        )
        
        # Transform both points to pixel coordinates
        x_center, y_center = wcs.world_to_pixel(sky_center)
        x_offset, y_offset = wcs.world_to_pixel(sky_offset)
        
        # Calculate image PA (Right of Top)
        # In image coordinates: +X is right, +Y is up (in standard matplotlib display)
        dx = x_offset - x_center
        dy = y_offset - y_center
        
        # arctan2(dx, dy) gives angle Right of Top
        pa_image = np.degrees(np.arctan2(dx, dy))
        
        return pa_image

    @classmethod
    def plot_pa_on_image(cls, image, x, y, pa_east_of_north, wcs, length=50, 
                        ax=None, color='red', linewidth=2, label=None):
        """
        Plot a position angle line on an astronomical image.
        
        Parameters
        ----------
        image : 2D array
            Image data to display
        x : float
            X pixel coordinate (center of PA line)
        y : float
            Y pixel coordinate (center of PA line)
        pa_east_of_north : float
            Position angle in degrees (East of North)
        wcs : astropy.wcs.WCS
            WCS object for the image
        length : float, optional
            Length of PA line in pixels (default: 50)
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new figure if None)
        color : str, optional
            Color of PA line (default: 'red')
        linewidth : float, optional
            Width of PA line (default: 2)
        label : str, optional
            Label for the PA line
        
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display the image
        ax.imshow(image, origin='lower', cmap='gray', interpolation='nearest')
        
        # Convert PA to image coordinates
        pa_image = cls.pa_sky_to_image(x, y, pa_east_of_north, wcs)
        
        # Calculate line endpoints in image coordinates
        pa_rad = np.radians(pa_image)
        dx = length/2 * np.sin(pa_rad)
        dy = length/2 * np.cos(pa_rad)
        
        x1, y1 = x - dx, y - dy
        x2, y2 = x + dx, y + dy
        
        # Plot the PA line
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                label=label if label else f'PA = {pa_east_of_north:.1f}° E of N')
        
        # Mark the center point
        ax.plot(x, y, 'o', color=color, markersize=8)
        
        # Add arrow at the positive direction
        ax.annotate('', xy=(x2, y2), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=linewidth))
        
        return ax

    @classmethod
    def plot_compass_rose(cls, ax, x, y, wcs, size=30, color='cyan'):
        """
        Add a compass rose showing North and East directions on the image.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        x : float
            X pixel coordinate for compass center
        y : float
            Y pixel coordinate for compass center
        wcs : astropy.wcs.WCS
            WCS object
        size : float, optional
            Size of compass arrows in pixels (default: 30)
        color : str, optional
            Color of compass arrows (default: 'cyan')
        """
        # North direction (PA = 0°)
        pa_north = cls.pa_sky_to_image(x, y, 0, wcs)
        dx_n = size * np.sin(np.radians(pa_north))
        dy_n = size * np.cos(np.radians(pa_north))
        
        # East direction (PA = 90°)
        pa_east = cls.pa_sky_to_image(x, y, 90, wcs)
        dx_e = size * np.sin(np.radians(pa_east))
        dy_e = size * np.cos(np.radians(pa_east))
        
        # Plot North arrow
        ax.annotate('',xy=(x + dx_n, y + dy_n), xytext=(x, y),
                    arrowprops=dict(
                    arrowstyle='-|>',
                    lw=2,
                    color=color,
                    mutation_scale=5,
                    shrinkA=0,
                    shrinkB=0))
        ax.text(x + dx_n*1.2, y + dy_n*1.2, 'N', color=color, 
                fontsize=7, fontweight='bold', ha='center', va='center')
        
        # Plot East arrow
        ax.annotate('',xy=(x + dx_e, y + dy_e), xytext=(x, y),
                    arrowprops=dict(
                    arrowstyle='-|>',
                    lw=2,
                    color=color,
                    mutation_scale=5,
                    shrinkA=0,
                    shrinkB=0))
        ax.text(x + dx_e*1.2, y + dy_e*1.2, 'E', color=color,
                fontsize=7, fontweight='bold', ha='center', va='center')
        
    @classmethod
    def plot_scale(cls, ax, x, y, wcs, size=30, color='cyan'):
        """
        Add a compass rose showing North and East directions on the image.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        x : float
            X pixel coordinate for compass center
        y : float
            Y pixel coordinate for compass center
        wcs : astropy.wcs.WCS
            WCS object
        size : float, optional
            Size of compass arrows in arcseconds (default: 30)
        color : str, optional
            Color of compass arrows (default: 'cyan')
        """
        
        pixel_scale = cls.get_pixel_scale(wcs.to_header())
        size_pix = size / pixel_scale
        
        if size < 60:
            scale_text =  f'{size}'+r'$^{\prime\prime}$'
        elif (size >= 60) and (size < 3600):
            scale_text =  f'{size/60:.1f}'+r'$^\prime$'
        else:
            scale_text =  f'{size/3600:.1f}'+r'$^\circ$'
        
        ax.annotate('',xy=(x + size_pix/2, y), xytext=(x - size_pix/2, y),
                    arrowprops=dict(
                    arrowstyle='-',
                    lw=2,
                    color=color,
                    mutation_scale=5,
                    shrinkA=0,
                    shrinkB=0))
        ax.text(x, y + size_pix * 0.3, scale_text, color=color,
                fontsize=7, fontweight='bold', ha='center', va='center')
        
        
    @classmethod
    def update_header(cls, ori_header, new_header):
        """
        Update original header with new header information.
        Preserves all original header information and replaces/adds values from new_header.
        
        Parameters
        ----------
        ori_header : astropy.io.fits.Header
            Original header to be updated
        new_header : astropy.io.fits.Header
            Header containing new/updated values
        
        Returns
        -------
        astropy.io.fits.Header
            Updated original header with new values merged in
        """
        # Create a copy to avoid modifying the original
        matrix_keys = ["CD", "PC", "CDELT", "PV"]
        
        updated_header = ori_header.copy()
        
        # Update with new header values
        for key in new_header:
            if (key != "COMMENT") or any(matrix in key for matrix in matrix_keys):
                updated_header[key] = new_header[key]
        
        return updated_header