import numpy as np
from scipy.stats import mode
from importlib import resources
import warnings
import sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.utils.exceptions import AstropyWarning

from photutils.detection import DAOStarFinder
from photutils.psf import extract_stars, EPSFBuilder, EPSFFitter, fit_2dgaussian

warnings.simplefilter('ignore', category=AstropyWarning)

from ..utils.utility import useful_functions
from ..handlers.filter_handler import Filters
from ..utils.file_handler import Parsers

class PointSpreadFunction:
    filt_inst = Filters()
    
    def __call__(self):
        pass
    
    @classmethod
    def extract(cls, image_data, header, galaxy_name, observatory, band, image_set):
        fwhm_val = cls.get_epsf(image_data, header, galaxy_name, observatory, band)
        
        if fwhm_val is None:
            fwhm_val = cls.measure_psf_fwhm(cls, image_data, header, threshold_sigma=15)
            if fwhm_val == -1:
                obs_psfs = useful_functions.extract_values_recursive(image_set.psf[galaxy_name], observatory)
                obs_psfs = list(filter(lambda x: isinstance(x, float), obs_psfs))
                fwhm_val = np.nanmedian(obs_psfs)
        else:
            fwhm_val = fwhm_val.item() if isinstance(fwhm_val, np.ndarray) else float(fwhm_val)
        
        image_set.psf = (galaxy_name, observatory, band, fwhm_val) # in ", fwhm * pixel_scale
    
    @classmethod
    def convolution(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set):
        """
        Convolve `image` with a Gaussian kernel of width `sigma_extra_pix` (pixels).
        If sigma_extra_pix==0, return original image.
        """
        psf_list = useful_functions.extract_values_recursive(image_set.psf, galaxy_name)
        
        sig_i = image_set.psf[galaxy_name][observatory][band]
        sig_t = np.nanmax(psf_list)
        sig_3 = np.nanmedian(psf_list) + 3 * np.nanstd(psf_list)
        sig_t = np.min([sig_t, sig_3])
        
        pixel_scale = useful_functions.get_pixel_scale(header)
        del_psf = np.sqrt((sig_t/2.355)**2 - (sig_i/2.355)**2) / pixel_scale

        if (del_psf <= 0) or (np.isnan(del_psf)):
            return

        kernel = Gaussian2DKernel(x_stddev=del_psf)

        # Try GPU acceleration
        use_gpu = False
        try:
            import cupy as cp
            from cupyx.scipy.signal import fftconvolve
            use_gpu = True
        except ImportError:
            pass

        if use_gpu:
            try:
                # Prepare kernel
                k_arr = kernel.array.astype(np.float32)
                # Normalize kernel
                k_arr = k_arr / k_arr.sum()
                k_gpu = cp.asarray(k_arr)

                def _gpu_convolve(data, k_gpu):
                    d_gpu = cp.asarray(data, dtype=np.float32)
                    nan_mask = cp.isnan(d_gpu)
                    
                    if cp.any(nan_mask):
                        # Normalized convolution for NaN handling
                        d_gpu[nan_mask] = 0.0
                        w_gpu = 1.0 - nan_mask.astype(np.float32)
                        
                        conv_d = fftconvolve(d_gpu, k_gpu, mode='same')
                        conv_w = fftconvolve(w_gpu, k_gpu, mode='same')
                        
                        # Avoid division by zero
                        conv_w[conv_w < 1e-9] = 1.0
                        result = conv_d / conv_w
                    else:
                        result = fftconvolve(d_gpu, k_gpu, mode='same')
                    
                    return cp.asnumpy(result)

                convolved_img = _gpu_convolve(image_data, k_gpu)
                convolved_err = _gpu_convolve(error_data, k_gpu)
                
            except Exception as e:
                print(f"GPU convolution failed: {e}. Falling back to CPU.")
                use_gpu = False

        if not use_gpu:
            # Force kernel to float32
            kernel_arr = kernel.array.astype(np.float32)
            
            convolved_img = convolve_fft(
                        image_data.astype(np.float32), kernel_arr,
                        normalize_kernel=True,
                        nan_treatment='interpolate',
                        allow_huge=True
            ).astype(np.float32)
            convolved_err = convolve_fft(
                error_data.astype(np.float32), kernel_arr,
                normalize_kernel=True,
                nan_treatment='interpolate',
                allow_huge=True
            ).astype(np.float32)

        image_set.update_data(convolved_img, galaxy_name, observatory, band)
        image_set.update_error(convolved_err, galaxy_name, observatory, band)

        del convolved_img
        del convolved_err
    
    
    @classmethod
    def detect_star(cls, image, header, threshold_sigma, galaxy, observatory, band):
        curve = cls.filt_inst.get_filter(name=band, facility=observatory)
        mask = (curve.response != 0)
        wave = curve.wavelength[mask]
        min_wave = np.nanmin(wave)
        
        if (threshold_sigma < 10) or (min_wave > 12 * 1e4):
            stars = cls.get_predefined_model(observatory, band)
            return stars
        
        im_x, im_y = image.shape
        mean, median, std = sigma_clipped_stats(image, sigma=10.0)

        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma*std)
        sources = daofind(image)

        if sources is None:
            daofind = DAOStarFinder(fwhm=15.0, threshold=threshold_sigma*std/5)
            sources = daofind(image)
            
            if sources is None:
                print("No sources detected even in lose criteria.")
                return None

        size = int(50 / useful_functions.get_pixel_scale(header)) // 2 * 2 + 1
        hsize = (size - 1) / 2
        gal_coord = useful_functions.get_sky_loc(galaxy)
        x_gal, y_gal = WCS(header).world_to_pixel(gal_coord)

        x, y = sources['xcentroid'], sources['ycentroid']
        mask = ((x > hsize) & (x < (image.shape[1] -1 - hsize)) &
                (y > hsize) & (y < (image.shape[0] -1 - hsize)) &
                (np.sqrt((x-x_gal)**2 + (y-y_gal)**2) > (10 * size)))

        stars_tbl = Table()
        stars_tbl['x'] = x[mask]
        stars_tbl['y'] = y[mask]

        nddata = NDData(data=image)

        stars = extract_stars(nddata, stars_tbl, size=size)

        if len(stars) > 20:
            return stars
        else:
            return cls.detect_star(image, header, threshold_sigma * 0.8, galaxy, observatory, band)


    @classmethod
    def get_epsf(cls, image=None, header=None, galaxy=None, observatory=None, band=None):
        
        image = np.nan_to_num(image, nan=0.0)

        stars = cls.detect_star(image=image, header=header, threshold_sigma=200, 
                                galaxy=galaxy, observatory=observatory, band=band)
        
        psf = None
        if not isinstance(stars, str):
            try:
                epsf_builder = EPSFBuilder(oversampling=1, maxiters=10, progress_bar=False, smoothing_kernel="quartic", fitter=EPSFFitter(fit_boxsize=7))
                epsf, fitted_stars = epsf_builder(stars)
                psf = epsf.data
            except:
                stars = cls.get_predefined_model(observatory, band)

        if psf is None:
            if stars is None:
                return None
            try:
                psf = fits.getdata(stars, ext=1)
                header = fits.getheader(stars, ext=1)
            except:
                psf = fits.getdata(stars)
                header = fits.getheader(stars)
        
        psf_center = (psf.shape[0]/2, psf.shape[1]/2)
        if psf_center[0] > 200:
            new_size = int(psf.shape[0] * 0.15)
            if new_size % 2 == 0:
                new_size += 1
            
            half_size = new_size // 2
            cy, cx = int(psf_center[0]), int(psf_center[1])
            
            psf = psf[cy - half_size : cy + half_size + 1, 
                      cx - half_size : cx + half_size + 1]
            psf_center = (psf.shape[0]/2, psf.shape[1]/2)
            
        fit_2d = fit_2dgaussian(psf, xypos=psf_center, fix_fwhm=False)
        fwhm = fit_2d.results["fwhm_fit"].value
        
        pixel_scale = useful_functions.get_pixel_scale(header)
        
        return fwhm * pixel_scale
    
    
    @classmethod
    def get_predefined_model(cls, observatory, band):
        filter_dir = resources.files("Spec7DT.reference.psfs")
        for filepath in filter_dir.iterdir():
            if filepath.name.endswith('.fits'):
                try:
                    with resources.as_file(filepath) as file_path:
                        file_path = str(file_path)
                        obs_file = Parsers._observatory_name_parser(file_path)
                        band_file = Parsers._band_name_parser(file_path, cls.filt_inst)
                        if (obs_file.lower() == observatory.lower()) & (band_file.lower() == band.lower()):
                            print("Load Pre-defined PSF model.")
                            return file_path
                        else:
                            continue
                except:
                    continue
        

    def measure_fwhm_gaussian(image, x_center, y_center, box_size=21):
        """
        Measure FWHM by fitting 2D Gaussian to PSF
        """
        # Extract cutout around star
        y, x = np.ogrid[:box_size, :box_size]
        x_start = int(x_center - box_size//2)
        y_start = int(y_center - box_size//2)
        
        cutout = image[y_start:y_start+box_size, x_start:x_start+box_size]
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[:box_size, :box_size]
        
        # Initial parameter guess
        amplitude = np.max(cutout)
        x_mean = box_size // 2
        y_mean = box_size // 2
        
        # Fit 2D Gaussian
        g_init = models.Gaussian2D(amplitude=amplitude, 
                                x_mean=x_mean, y_mean=y_mean,
                                x_stddev=box_size * 0.05, y_stddev=box_size * 0.05)
        fit_g = fitting.TRFLSQFitter()
        g = fit_g(g_init, x_grid, y_grid, cutout)
        
        # Convert stddev to FWHM
        fwhm_x = 2.355 * g.x_stddev.value
        fwhm_y = 2.355 * g.y_stddev.value
        fwhm_avg = (fwhm_x + fwhm_y) / 2
        
        return fwhm_avg

    def measure_psf_fwhm(self, image, header, threshold_sigma=15):
        """
        Complete pipeline: detect stars and measure FWHM
        """
        im_x, im_y = image.shape
        box_size = int((im_x + im_y) * 0.1 / 2)
        mean, median, std = sigma_clipped_stats(image, sigma=10.0)
        
        # Star detection
        daofind = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma*std)
        sources = daofind(image)
        
        if sources is None:
            daofind = DAOStarFinder(fwhm=15.0, threshold=threshold_sigma*std/5)
            sources = daofind(image)
            
            if sources is None:    
                print("No sources detected even in lose criteria. Return -1")
                return -1.0
        
        # Measure FWHM for each detected star
        fwhm_measurements = []
        
        for source in sources:
            x, y = source['xcentroid'], source['ycentroid']
            
            margin = 0.1
            # Skip stars too close to edges
            if (x > im_x * margin and x < im_x * (1 - margin) and 
                y > im_y * margin and y < im_y * (1 - margin)):
                
                try:
                    fwhm = self.measure_fwhm_gaussian(image, x, y, box_size=box_size)
                    if not np.isnan(fwhm) and fwhm > 0:
                        fwhm_measurements.append(fwhm)
                except:
                    continue
        
        fwhm, _ = mode(fwhm_measurements, nan_policy='omit')
        cd_mx = np.abs(header.get("CD1_1", 1.0)) * 3600  # in "
        cdelt_mx = np.abs(header.get("CDELT1", 1.0)) * 3600
        pc_mx = np.abs(header.get("PC1_1", 1.0)) * 3600
        
        if (cd_mx > 3000) and (cdelt_mx > 3000) and (pc_mx > 3000):
            raise ValueError("No valid WCS matrix in header.")
        
        pixel_scale = min(cd_mx, cdelt_mx, pc_mx)
        
        return fwhm * pixel_scale
