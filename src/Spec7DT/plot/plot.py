import matplotlib.pyplot as plt
import matplotlib.axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from astropy.visualization import (
    AsinhStretch, AsymmetricPercentileInterval,
    ImageNormalize,
)
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy.stats import percentileofscore

# from ..utils.file_handler import GalaxyImageSet
from ..utils.utility import useful_functions


plt.rcParams["font.family"] = "FreeSerif"

plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7


class DrawGalaxy:
    def __init__(self):
        pass
    
    @classmethod
    def plot_galaxies(cls, image_set, galaxy: str, step: str):
        image_dict = {}
        
        galaxy_data = image_set.data[galaxy]
        
        for obs, obs_dict in galaxy_data.items():
            for band, band_im in obs_dict.items():
                image_dict[f"{obs}.{band}"] = band_im
        
             
        m, n = useful_functions.find_rec(len(image_dict))
        
        fig, axes = plt.subplots(m, n, dpi=200, figsize=(n * 1.5, m * 1.2))
        
        for ax, im_data in zip(axes.flatten(), image_dict.items()):
            norm = ImageNormalize(im_data[1], interval=AsymmetricPercentileInterval(50., 99.8), stretch=AsinhStretch())
            
            im = ax.imshow(im_data[1], cmap='gray', origin="lower", norm=norm)
            
            ax.tick_params(axis="both", which="both", direction="in")
            ax.tick_params(axis="both", which="major", width=1.2)
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            
        for ax in axes.flatten()[len(image_dict):]:
            ax.remove()
        
        fig.suptitle(f"Step Name: {step}")
        fig.tight_layout()
        plt.show()
    
    @classmethod
    def single_galaxy(cls, image_set, galaxy: str, obs: str, band: str, compass=True, scale=True):
        im_data = image_set.data[galaxy][obs][band]
        im_hdr = image_set.header[galaxy][obs][band]
        x, y = im_data.shape
        
        fig, ax = plt.subplots(1, 1, dpi=200, figsize=(4, 4), subplot_kw=dict(projection=WCS(im_hdr)))
            
        _, median, _ = sigma_clipped_stats(im_data, sigma=10.0, mask=np.where(im_data == 0, True, False))
        bg_percent = percentileofscore(np.nan_to_num(im_data.flatten()), median, kind="mean")
        
        norm = ImageNormalize(im_data, interval=AsymmetricPercentileInterval(bg_percent, 99.8), stretch=AsinhStretch())
        
        im = ax.imshow(im_data, cmap='gray', origin="lower", norm=norm)
        
        if compass:
            useful_functions.plot_compass_rose(ax, x * 0.9, y * 0.1, WCS(im_hdr), size=1/12*x, color='white')
            
        if scale:
            useful_functions.plot_scale(ax, x * 0.1, y * 0.1, WCS(im_hdr), size=120, color='white')
        
        ax.tick_params(axis="both", which="both", direction="in", color="#CCC")
        ax.tick_params(axis="both", which="major", width=1.2)
        ax.set_xlabel(r"$\alpha_{2000}$")
        ax.set_ylabel(r"$\delta_{2000}$")
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=matplotlib.axes.Axes)
        cax.tick_params(axis="y", which="both", direction="in", color="#000")
        fig.colorbar(im, cax=cax, orientation='vertical', label="mJy")
        
        # ax.set_title(f"{galaxy} {obs} {band}")
        fig.tight_layout()
        
        plt.show()
        
        return fig, ax