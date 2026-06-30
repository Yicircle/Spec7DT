try:
    from .data_loader import CatalogueManager
    from .radial_profile import radial_property
    from .utility import GalaxyUtils, TypeParser, Parsers, UnitConverter
    from .consts import PlotConfig
except ImportError:
    from data_loader import CatalogueManager
    from radial_profile import radial_property
    from utility import GalaxyUtils, TypeParser, Parsers, UnitConverter
    from consts import PlotConfig

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import seaborn as sns
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from pyvo.dal.exceptions import DALServiceError
from requests.exceptions import ConnectionError

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams['mathtext.fontset'] = 'stix'


class GalaxyProperty:
    def __init__(self):
        self.image_dict = {}
        self.dist = {}
        self.dist_red = {}
        
    def __dir__(self):
        return sorted(super().__dir__(), key=str.upper)
    
    
    def load_from_catalog(self, results_file, im_size, platescale, cat_name=None, supply_file=None):
        cat_manager = CatalogueManager()
        self.image_size = im_size
        self.platescale = platescale
        self.results_file, self.date_tag, self.results_dict, self.total, self.units = cat_manager.load(results_file, im_size, supply_file)
        
        obs_file = results_file.parent / "observations.fits"
        self.obs_file, _, self.obs_dict, self.obs_total, self.obs_units = cat_manager.load(obs_file, im_size)
        self.targets = list(self.results_dict.keys())
        self.filters = []
        
        self.name = str(Path(self.results_file).parent.name) if cat_name is None else cat_name.split(".")[-1]
        self.name = self.name.replace("_", " ")
        self.group = str(Path(self.results_file).parent.parent.name).replace("_", " ") if cat_name is None else cat_name.split(".")[0]
        
        for target in self.targets:
            self.image_dict[target] = {}
            
            try:
                d_Mpc = GalaxyUtils.query_distance(target)
                
            except (ValueError, DALServiceError, ConnectionError) as e:
                print(f"Warning: Failed to query distance for {target} from SIMBAD.")
                print(f"Reason: {e}")
                d_Mpc = None
                
            except Exception as e:
                print(f"Warning: Unexpected error querying {target}: {e}")
                d_Mpc = None
            self.dist[target] = d_Mpc
            
            try:
                z_s = self.results_dict[target]['best.universe.redshift']
                redshift = z_s[z_s != 0].mean()
                d_Mpc_redshift = GalaxyUtils.redshift_to_distance(redshift)
            except (ValueError, IndexError):
                d_Mpc_redshift = None
            self.dist_red[target] = round(d_Mpc_redshift, 6)
            
            # get all filters
            for ff in self.obs_dict[target].keys():
                if (ff not in ["id", "redshift", "distance"]) & ("_err" not in ff):
                    self.filters.append(ff)
        
        self.filters = list(set(self.filters))
        
        self.filter_prop, self.filter_resps = Parsers.get_filter_properties(self.filters)
        
        
        
    def load_image(self, key_names):
        if isinstance(key_names, str):
            key_names = [key_names]
        
        for key_name in key_names:
            if "age" in key_name:
                factor = 1e6  # yr
            else:
                factor = 1
                
            for target in self.targets:
                df = self.results_dict[target]
                im = df[key_name].values.reshape(self.image_size, self.image_size)
                self.image_dict[target][key_name] = im * factor
                
                if key_name+"_err" in df.keys():
                    err_im = df[key_name+"_err"].values.reshape(self.image_size, self.image_size)
                    self.image_dict[target][key_name+"_err"] = err_im * factor
                
    
    def get_image(self, target, key_name):
        if not key_name in self.image_dict[target]:
            self.load_image(key_name)
        
        return self.image_dict[target][key_name]
        

    def _get_wcs(self, wcs_header):
        if isinstance(wcs_header, WCS):
            return wcs_header
        if isinstance(wcs_header, (str, Path)):
            return WCS(fits.getheader(wcs_header))
        return WCS(wcs_header)
    
    def _make_fov_mask(self, shape, wcs_header, fov_vertices):
        if wcs_header is None or fov_vertices is None:
            raise ValueError("Both wcs_header and fov_vertices are required for FOV statistics.")
        
        wcs = self._get_wcs(wcs_header)
        
        if isinstance(fov_vertices, SkyCoord):
            coords = fov_vertices.icrs
        else:
            vertices = np.asarray(fov_vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
                raise ValueError("fov_vertices must be an array-like object with shape (N, 2).")
            coords = SkyCoord(vertices[:, 0], vertices[:, 1], unit="deg", frame="icrs")
        
        x_vertices, y_vertices = wcs.world_to_pixel(coords)
        pixel_vertices = np.column_stack((x_vertices, y_vertices))
        
        if not np.all(np.isfinite(pixel_vertices)):
            raise ValueError("FOV vertices could not be converted to finite pixel coordinates.")
        
        y_grid, x_grid = np.mgrid[:shape[0], :shape[1]]
        pixel_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        fov_path = MplPath(pixel_vertices)
        
        return fov_path.contains_points(pixel_points, radius=0.5).reshape(shape)
    
    def get_fov_property_stats(self, target, property_name, wcs_header, fov_vertices, ptype=None, z_sol=0.02):
        im = np.asarray(self.get_image(target, property_name), dtype=float)
        mask = self._make_fov_mask(im.shape, wcs_header, fov_vertices)
        values = im[mask]
        
        valid = np.isfinite(values) & (values != 0.0)
        if ptype is None:
            ptype = TypeParser.parse_type(property_name, unit_dict=self.units)
        
        if ptype in ["age", "metallicity"]:
            valid &= values > 0.0
        
        values = values[valid]
        if values.size == 0:
            raise ValueError(f"No valid pixels found inside the FOV for {target}: {property_name}.")
        
        if ptype == "age":
            stat_values = np.log10(values)
        elif ptype == "metallicity":
            stat_values = np.log10(values / z_sol)
        else:
            stat_values = values
        
        median = np.nanmedian(stat_values)
        std = np.nanstd(stat_values, ddof=1) if stat_values.size > 1 else 0.0
        
        return median, std
        
    def get_radial_profile(self, target=None, key_name=None, max_radius=None, im=None, **kwargs):
        radial_kwargs = dict(kwargs)
        im = self.get_image(target, key_name) if im is None else im

        center = (im.shape[1] / 2, im.shape[0] / 2)
        if "err_im" not in radial_kwargs:
            if target is not None and key_name is not None and key_name + "_err" in self.results_dict[target].keys():
                err_im = self.get_image(target, key_name + "_err")
            else:
                err_im = None
        else:
            err_im = radial_kwargs.pop("err_im")

        weight_arg = radial_kwargs.pop("weight", None)
        if isinstance(weight_arg, str):
            if target is not None and weight_arg in self.results_dict[target].keys():
                weight = self.get_image(target, weight_arg)
            else:
                weight = None
            if target is not None and weight_arg + "_err" in self.results_dict[target].keys():
                weight_err = self.get_image(target, weight_arg + "_err")
            else:
                weight_err = None
        elif isinstance(weight_arg, (list, tuple, np.ndarray, pd.Series)):
            weight = weight_arg
            weight_err = None
        else:
            weight = weight_err = None

        theta = radial_kwargs.pop("theta", -76.38)
        n_steps = radial_kwargs.pop("n_steps", 30)
        min_valid_frac = radial_kwargs.pop("min_valid_frac", 4 / 5)

        return radial_property(
            image=im,
            center=center,
            max_radius=max_radius,
            platescale=self.platescale,
            error_image=err_im,
            weight=weight,
            weight_err=weight_err,
            theta=theta,
            n_steps=n_steps,
            min_valid_frac=min_valid_frac,
            **radial_kwargs,
        )
    
    
    def plot_radial_profile(self, ax, target, property_name, max_radius, reduce_value, convert, color, label):
        radii, mean, err = self.get_radial_profile(target, property_name, max_radius)
        if convert:
            pass

        ax.fill_between(radii / reduce_value, mean-err/2, mean+err/2, color=color, alpha=0.3) if err is not None else None
        ax.plot(radii / reduce_value, mean, color=color, label=label)
        

    def compare_total(self, ax, target, property_name, compares: list, colors: list):
        ref_points = {
            "mass": 10 ** 10.84 * (self.dist[target] / 11.32) ** 2,
            "luminosity": 0,
            "age": 9.83,
            "sfr": 10 ** 0.58,
            "metallicity": -0.01
        }
        
        ref_err = {
            "age": 0.0414,
            "metallicity": 0.0334,
            "sfr": 0.88
        }
        
        ref_label = {
            "mass": "Leroy+2021",
            "luminosity": 0,
            "age": "Optical Spectra (Pessa+2023)",
            "sfr": "CO (Leroy+2021)",
            "metallicity": "Optical Spectra (Pessa+2023)"
        }
        
        ylabels = PlotConfig.ylabels
        colors = sns.color_palette("hls", len(compares) + 1) if colors is None else colors
        
        ptype = TypeParser.parse_type(property_name, unit_dict=self.units)
        
        dist_factor = (self.dist[target] / self.dist_red[target]) ** 2 if ptype in ["mass", "luminosity", "sfr"] else 1
        y_values = [self.total[property_name].values[0] * dist_factor]
        y_errs = [self.total[property_name+"_err"].values[0] * dist_factor] if property_name+"_err" in self.total else [0]
        x_ticks = [self.name]
        
        alternates = [(i+1) % 2 for i in range(len(compares))]
        
        for i, compare in enumerate(compares):
            y_values.append(compare.total[property_name].values[0] * dist_factor)
            y_errs.append(compare.total[property_name+"_err"].values[0] * dist_factor) if property_name+"_err" in compare.total else y_errs.append(0)
            x_ticks.append("\n" * alternates[i]+compare.name)
        y_values = np.array(y_values)
        y_errs = np.array(y_errs)
        
        conv = UnitConverter()
        if ptype == "metallicity":
            y_values, y_errs = conv.from_metallicity(y_values, y_errs)
        elif ptype == "age":
            y_values, y_errs = conv.from_age(y_values, y_errs)
        
        x_values = list(np.arange(0, len(compares)+1, 1))
        
        if ptype in ref_points:
            if ptype in ref_err:
                ax.axhspan(ref_points[ptype]-ref_err[ptype], ref_points[ptype]+ref_err[ptype], alpha=0.15, color='#333')
            ax.axhline(ref_points[ptype], ls="-", lw=1.5, color="#333", label=ref_label[ptype])
        ax.scatter(x_values, y_values, c=colors, marker=".", s=50)
        
        for x_value, y_value, y_err, color in zip(x_values, y_values, y_errs, colors):
            ax.errorbar(x_value, y_value, y_err, color=color, alpha=1, capsize=10)
        ax.set_ylabel(ylabels.get(ptype, "Y"))
        ax.set_xticks(ticks=x_values, labels=x_ticks)
        
        return ax
