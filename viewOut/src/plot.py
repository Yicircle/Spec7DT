from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
import seaborn as sns
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

from cmap import Colormap
import copy
import pandas as pd

try:
    from .utility import TypeParser, Parsers, UnitConverter
    from .consts import Colors, Physicals, PlotConfig
except ImportError:
    from utility import TypeParser, Parsers, UnitConverter
    from consts import Colors, Physicals, PlotConfig

plt.rcParams["font.family"] = "Liberation Serif"
plt.rcParams['mathtext.fontset'] = 'stix'


class PlotProperty:
    def __init__(self):
        pass

    @classmethod
    def get_filter_inst(cls):
        return Parsers.get_filter_inst()


    @classmethod
    def plot_property(cls, ax, property_name, plot_ref=True, units=None, z_sol=0.01524):
        
        ptype = TypeParser.parse_type(property_name, unit_dict=units)
        
        if (ptype in PlotConfig.refer_values) & (plot_ref):
            if ptype == "age":
                _err = 0.0414
            elif ptype == "metallicity":
                _err = 0.0334
            else:
                _err = 0
            # ax.fill_between(PlotConfig.refer_values[ptype][0], np.array(PlotConfig.refer_values[ptype][1])-_err, np.array(PlotConfig.refer_values[ptype][1])+_err, color=Colors.defaults.charcoal, alpha=0.05)
            x, y = np.array(PlotConfig.refer_values[ptype][0]), np.array(PlotConfig.refer_values[ptype][1])
            y = np.log10(np.power(10, y) * 0.019 / z_sol) if ptype == "metallicity" else y
            ax.plot(x, y, c=Colors.defaults.charcoal, ls="-", label="Optical Spectra (Pessa+2023)")
        
        if (ptype in PlotConfig.refer_values2) & (plot_ref):
            # ax.fill_between(PlotConfig.refer_values2[ptype][0], np.array(PlotConfig.refer_values2[ptype][1])-0.1, np.array(PlotConfig.refer_values2[ptype][1])+0.1, color=Colors.defaults.outer_space, alpha=0.05)
            x2, y2 = np.array(PlotConfig.refer_values2[ptype][0]), np.array(PlotConfig.refer_values2[ptype][1])
            y2 = np.log10(np.power(10, y2) * 0.019 / z_sol) if ptype == "metallicity" else y2
            ax.plot(x2, y2, c=Colors.defaults.outer_space, ls="--", label="FUV-FIR Photometry (Abdurro’uf+2022)")
            # pass
        
        ax.set_xlabel("Radius "+r"[$R/R_{25}$]")
        ax.set_ylabel(PlotConfig.ylabels.get(ptype, "Y"))
        ax.tick_params(axis="both", which="both", direction="in")
        ax.tick_params(axis="both", which="major", width=1.2)
        ax.grid(visible=True, which="major", axis="both",
                color=Colors.defaults.gray, alpha=0.3, linewidth=0.5, linestyle="--")
        
        if ptype == 'age':
            ax.set_ylim(7.7, 10.2)
        elif ptype == 'metallicity':
            ax.set_ylim(-0.6, 0.21)
        
    @classmethod
    def plot_sed(cls, ax, galaxy_property, galaxy, idx, margin: tuple=(0.15, 0.6)):
        if idx != "Total":
            dist_factor = galaxy_property.results_dict[galaxy].iloc[idx]["best.universe.luminosity_distance"]
        else:
            dist_factor = galaxy_property.total["best.universe.luminosity_distance"].values[0]
            
        
        # dist_factor = galaxy_property.dist[galaxy] * 1e6 * 3.086e16
        area_factor = 4.0 * np.pi * dist_factor ** 2
        c = 2.9979246e17  # in [nm/s]
        
        # Observation SED
        if idx != "Total":
            obs_flux = galaxy_property.obs_dict[galaxy].iloc[idx]
        else:
            obs_flux = galaxy_property.obs_total.iloc[0]
        center_waves = galaxy_property.filter_prop["pivot"]
        fwhms = galaxy_property.filter_prop["FWHM"]
        
        # Results SED
        if idx != "Total":
            sed_file = galaxy_property.results_file.parent / f"{galaxy}_{idx}_best_model.fits"
        else:
            sed_file = galaxy_property.results_file.parent / f"{idx}_best_model.fits"
        
        try:
            sed_table = Table.read(sed_file, format="fits").to_pandas()
            sed_header = fits.getheader(sed_file, ext=1)
        except Exception as e:
            return None, None, None

        A_v = float(sed_header["attenuation.generic.bessell.V"])
        metal_star = float(sed_header["stellar.metallicity"])
        age_star = float(sed_header["sfh.age"]) * 1e-3
        
        property_text = f"Stellar Age: {age_star:.1f} [Gyr]"+"\n"+rf"Stellar Metallicity: {metal_star} [$Z_{{\odot}}$]"+"\n"+rf"$A_{{V}}$: {A_v:.2f} [mag]"
        
        
        factor = 1e29 / c / area_factor
        
        for col in sed_table.columns[2:]:
            sed_table[col] *= sed_table["wavelength"] ** 2 * factor

        sed_table["wavelength"] *= 1e-3
        
        stellar_unatt = sed_table["stellar.old"] + sed_table["stellar.young"]
        stellar_att = stellar_unatt + sed_table["attenuation.stellar.old"] + sed_table["attenuation.stellar.young"]
        nebular = sed_table["nebular.absorption_old"] + sed_table["nebular.absorption_young"] + sed_table["nebular.emission_old"] + sed_table["nebular.emission_young"]
        nebular = nebular + sed_table["attenuation.nebular.emission_old"] + sed_table["attenuation.nebular.emission_young"]
        dust = sed_table["dust.Umin_Umin"] + sed_table["dust.Umin_Umax"]
        
        phot_wave = []
        phot_flux = []
        phot_err = []
        for i, (key, flux) in enumerate(obs_flux.items()):
            if (key not in ["id", "redshift", "distance"]) & ("_err" not in key):
                _obs = Parsers._observatory_name_parser(key)
                _band = Parsers._band_name_parser(key, cls.get_filter_inst())
                flux_err = obs_flux[f"{key}_err"]
                phot_wave.append(center_waves[f"{_obs}.{_band}"] * 1e-4); phot_flux.append(flux); phot_err.append(flux_err)
            else:
                continue

        if ax is None:
            phots = (np.array(phot_wave), np.array(phot_flux), np.array(phot_err))
            models = (np.array(sed_table["wavelength"]), sed_table["L_lambda_total"])
            comps = (np.array(stellar_att), np.array(nebular), np.array(dust))
            return phots, models, comps
        else:
            # Plot SEDs
            ax.plot(sed_table["wavelength"], stellar_unatt, c=Colors.defaults.amber, ls="--", alpha=0.7, label="Un-attenuated Stellar")
            ax.plot(sed_table["wavelength"], stellar_att, c=Colors.defaults.amber, label="Attenuated Stellar", lw=0.7)
            
            ax.plot(sed_table["wavelength"], nebular, c=Colors.defaults.green, label="Nebular", lw=0.7)
            ax.plot(sed_table["wavelength"], dust, c=Colors.defaults.dust_red, label="Dust", lw=0.7)
            ax.plot(sed_table["wavelength"], sed_table["L_lambda_total"], c=Colors.defaults.dark_gray, lw=1.2, label="Best SED Model")
            
            # Plot Obs SEDs
            
            for i, (key, flux) in enumerate(obs_flux.items()):
                if (key not in ["id", "redshift", "distance"]) & ("_err" not in key):
                    _obs = Parsers._observatory_name_parser(key)
                    _band = Parsers._band_name_parser(key, cls.get_filter_inst())
                    temp_df = obs_flux.reset_index(drop=True)
                    flux_err = obs_flux[f"{key}_err"] if (obs_flux[f"{key}_err"] < (0.5 * obs_flux[f"{key}"])) & (obs_flux[f"{key}_err"] > 0) else [0]
                else:
                    continue
                
                label = "Observed Fluxes" if i == 1 else None
                
                ax.scatter(center_waves[f"{_obs}.{_band}"] * 1e-4, flux, s=20, facecolors='none', edgecolors=Colors.defaults.obs_edge)
                ax.errorbar(
                    center_waves[f"{_obs}.{_band}"] * 1e-4, flux, yerr=flux_err, xerr=fwhms[f"{_obs}.{_band}"] * 5e-5,
                    marker='o', lw=0, elinewidth=1, capsize=2,
                    fillstyle="none", color=Colors.defaults.obs_color, ms=1, label=label
                    )
            
            ax.text(0.98, 0.98, property_text, fontsize=15, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
            
            ax.set_xscale("log")
            ax.set_yscale("log")
            

            # Set x/y axis margin
            phot_wave_arr = np.array(phot_wave)
            phot_flux_arr = np.array(phot_flux)
            
            valid_x_mask = phot_wave_arr > 0
            
            if np.any(valid_x_mask):
                x_min, x_max = phot_wave_arr[valid_x_mask].min(), phot_wave_arr[valid_x_mask].max()
                
                x_margin_factor = 10 ** margin[0] 
                x_lim_lower = x_min / x_margin_factor
                x_lim_upper = x_max * x_margin_factor
                ax.set_xlim(x_lim_lower, x_lim_upper)
                
                y_obs_mask = valid_x_mask & (phot_wave_arr >= x_lim_lower) & (phot_wave_arr <= x_lim_upper) & (phot_flux_arr > 0)
                y_obs_valid = phot_flux_arr[y_obs_mask]
                
                model_wave = np.array(sed_table["wavelength"])
                model_flux = np.array(sed_table["L_lambda_total"])
                y_model_mask = (model_wave >= x_lim_lower) & (model_wave <= x_lim_upper) & (model_flux > 0)
                y_model_valid = model_flux[y_model_mask]
                
                all_y = np.concatenate([y_obs_valid, y_model_valid])
                
                if len(all_y) > 0:
                    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
                    
                    y_margin_factor = 10 ** margin[1]
                    ax.set_ylim(y_min / y_margin_factor, y_max * y_margin_factor)
            
            ax.set_xlabel(r"Wavelength [$\mathrm{\mu}$m]")
            ax.set_ylabel(r"Flux $S_{\nu}$ [mJy]")
            
            ax.set_title(f"Best Model SED\n for Galaxy: {galaxy}, Pix ID: {idx}")
            
            ax.tick_params(axis="both", which="both", direction="in")
            ax.tick_params(axis="both", which="major", width=1.2)
            ax.grid(visible=True, which="major", axis="both",
                    color=Colors.defaults.gray, alpha=0.3, linewidth=0.5, linestyle="--")
            
            ax.legend(fontsize=13)
            
            phots = (np.array(phot_wave), np.array(phot_flux), np.array(phot_err))
            models = (np.array(sed_table["wavelength"]), sed_table["L_lambda_total"])
            comps = (np.array(stellar_att), np.array(nebular), np.array(dust))
            return phots, models, comps



    def plot_galaxy_properties(galaxy_instance, galaxy_name, wcs_header, width, property_list, patches=None):
        """
        GalaxyProperty 객체로부터 2D 물리 지도(fig) 패널을 생성합니다.
        - 배경 노이즈를 억제하고 은하 내부 구조를 강조하기 위해 Percentile 기반 자동 스케일링을 적용합니다.
        """
        ra, dec = SkyCoord.from_name(galaxy_name).ra.degree, SkyCoord.from_name(galaxy_name).dec.degree
        wcs = WCS(fits.getheader(wcs_header))
        
        meta = Physicals.params
        
        maps, labels, cmaps, vmins, vmaxs = [], [], [], [], []
        
        # [선택 사항] 배경 마스킹을 위한 기준 맵 (예: 첫 번째 property가 Luminosity나 Mass라고 가정)
        # 은하의 형태를 결정하는 기준 맵을 미리 계산하여 마스크로 활용할 수 있습니다.
        # ref_mask = None

        for idx, key in enumerate(property_list):
            if key not in meta:
                continue
            cfg = meta[key]
            raw_map = cfg["calc"](galaxy_instance, galaxy_name)
            
            # 0 이하의 값은 기본적으로 NaN 처리
            valid_map = np.where(raw_map <= 0, np.nan, raw_map)
            
            ref_shape = valid_map.shape
            center = int(ref_shape[0] / 2)
            
            # -------------------------------------------------------------
            # [핵심 개선] 화면에 표시될 중심 영역(Cropped box)만 추출하여 통계 계산
            # -------------------------------------------------------------
            y_start = int(max(0, center - width/2))
            y_end = int(min(ref_shape[0], center + width/2))
            x_start = int(max(0, center - width/2))
            x_end = int(min(ref_shape[1], center + width/2))
            
            cropped_view = valid_map[y_start:y_end, x_start:x_end]
            
            # 유효한(NaN이 아닌) 픽셀만 추출
            valid_pixels = cropped_view[np.isfinite(cropped_view)]
            
            if len(valid_pixels) > 0:
                # 배경 노이즈(하위 2%)와 튀는 픽셀(상위 99%)을 제외하여 vmin, vmax 동적 계산
                # Age나 Z_* 맵의 특성에 따라 이 퍼센타일 비율(예: 5, 95)을 조절하시면 더 타이트해집니다.
                auto_vmin = np.nanpercentile(valid_pixels, 2)
                auto_vmax = np.nanpercentile(valid_pixels, 99)
            else:
                auto_vmin, auto_vmax = cfg["vmin"], cfg["vmax"] # 예외 상황 시 기존 설정 폴백

            maps.append(valid_map)
            labels.append(cfg["label"])
            cmaps.append(cfg["cmap"])
            vmins.append(auto_vmin) # 자동 계산된 vmin 사용
            vmaxs.append(auto_vmax) # 자동 계산된 vmax 사용

        # --- Plotting 로직 ---
        fig, axes = plt.subplots(2, 3, subplot_kw=dict(projection=wcs))

        for i, ax in enumerate(axes.flatten()):
            if i >= len(maps):
                break
                
            # 수정된 vmin, vmax가 적용됨
            im = ax.imshow(maps[i], cmap=cmaps[i], origin="lower", vmin=vmins[i], vmax=vmaxs[i])
            
            div_m = make_axes_locatable(ax)
            cax_m = div_m.append_axes('right', size='5%', pad=0.05, axes_class=maxes.Axes)
            cbar = fig.colorbar(im, cax=cax_m, orientation='vertical')
            cbar.set_label(labels[i])
            cax_m.tick_params(axis="y")

            if patches is not None:
                for patch_obj in patches:
                    new_patch = copy.copy(patch_obj)
                    new_patch.set_transform(ax.get_transform('icrs'))
                    ax.add_patch(new_patch)

            ax.set_xlim(center - width, center + width)
            ax.set_ylim(center - width, center + width)
            
            # 배경색 설정 (NaN 처리된 부분은 이 색으로 표시됨)
            ax.set_facecolor(Colors.defaults.face_gray)
            
            ax.set_xlabel(r"$\alpha_{2000}$")
            ax.set_ylabel(r"$\delta_{2000}$")
            ax.tick_params(axis="both", which="both", direction="in", width=1.2)
            ax.grid(visible=True, which="major", color=Colors.defaults.gray, alpha=0.3, linewidth=0.5, linestyle="--")

            if i % 3 != 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False)
            if i < 3:
                ax.set_xticklabels([])
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)
            
        return fig

    @staticmethod
    def _select_wcs_header(wcs_header, index, g_data, target):
        if isinstance(wcs_header, (list, tuple)):
            return wcs_header[index]
        if isinstance(wcs_header, dict):
            return wcs_header.get(g_data.name, wcs_header.get(target, wcs_header.get("default")))
        return wcs_header

    @classmethod
    def compare_total(cls, ax, target, property_name, galaxy_data_list: list, colors: list = None, wcs_header=None, fov_vertices=None):
        """
        Compare total galaxy properties between different data sets.
        """
        if not galaxy_data_list:
            return ax

        base_data = galaxy_data_list[0]
        ylabels = PlotConfig.ylabels
        
        num_items = len(galaxy_data_list)
        colors = sns.color_palette("hls", num_items) if colors is None else colors
        
        ptype = TypeParser.parse_type(property_name, unit_dict=base_data.units)
        
        dist_factor = (base_data.dist[target] / base_data.dist_red[target]) ** 2 if ptype in ["mass", "luminosity", "sfr"] else 1
        use_fov_stats = wcs_header is not None or fov_vertices is not None
        if use_fov_stats and (wcs_header is None or fov_vertices is None):
            raise ValueError("Both wcs_header and fov_vertices are required for FOV statistics.")
        
        y_values = []
        y_errs = []
        x_ticks = []
        is_long_ticks = any(len(tick.name) > 7 for tick in galaxy_data_list)
        is_no_newline = all("\n" not in tick.name for tick in galaxy_data_list)
         
        for i, g_data in enumerate(galaxy_data_list):
            if use_fov_stats:
                selected_wcs_header = cls._select_wcs_header(wcs_header, i, g_data, target)
                if selected_wcs_header is None:
                    raise ValueError(f"No WCS header found for {g_data.name}: {target}.")
                y_value, y_err = g_data.get_fov_property_stats(
                    target, property_name, selected_wcs_header, fov_vertices, ptype=ptype
                )
                if ptype in ["mass", "luminosity", "sfr"]:
                    y_value *= dist_factor
                    y_err *= dist_factor
            else:
                y_value = g_data.total[property_name].values[0] * dist_factor
                y_err = g_data.total[property_name+"_err"].values[0] * dist_factor if property_name+"_err" in g_data.total else 0
            
            y_values.append(y_value)
            y_errs.append(y_err)
            x_ticks.append("\n" * (i % 2) + g_data.name) if is_long_ticks and is_no_newline else x_ticks.append(g_data.name)
            
        y_values = np.array(y_values)
        y_errs = np.array(y_errs)
        
        if not use_fov_stats:
            conv = UnitConverter()
            if ptype == "metallicity":
                y_values, y_errs = conv.from_metallicity(y_values, y_errs)
            elif ptype == "age":
                y_values, y_errs = conv.from_age(y_values, y_errs)
        
        x_values = list(np.arange(0, num_items, 1))
        
        # Plot reference values defined in PlotConfig.
        refs = PlotConfig.global_refs.get(ptype, [])
        for ref in refs:
            val = ref.get("value")
            # Calculate dynamic reference values when needed.
            if val is None and "value_calc" in ref:
                val = ref["value_calc"](base_data.dist[target])
                
            color, ls, label = ref.get("color", "#333"), ref.get("ls", "-"), ref.get("label", "")
            xmin, xmax = ref.get("xmin", 0.0), ref.get("xmax", 1.0)
            
            if "span" in ref and ref["span"]:
                span_lower, span_upper = ref["span"]
                ax.axhspan(span_lower, span_upper, xmin=xmin, xmax=xmax, alpha=0.15, color=color, label=label if val is None else None)
                
            if val is not None:
                ax.axhline(val, xmin=xmin, xmax=xmax, ls=ls, lw=1.5, color=color, label=label)
            
        ax.scatter(x_values, y_values, c=colors, marker=".", s=50)
        
        for x_value, y_value, y_err, color in zip(x_values, y_values, y_errs, colors):
            ax.errorbar(x_value, y_value, y_err, color=color, alpha=1, capsize=10)
            
        ax.set_ylabel(ylabels.get(ptype, "Y"))
        ax.set_xticks(ticks=x_values, labels=x_ticks)
        
        return ax

    @classmethod
    def plot_global_comparison(cls, galaxy_data_list: list, target: str, plot_props: list, out_path="/data9/wohylee/Figures/paper/global_prop.pdf", wcs_header=None, fov_vertices=None):
        """
        Generate a global properties comparison plot dynamically based on input galaxy data list and properties.
        """
        PlotConfig.set_paper_style("double")
        
        # Pre-defined colors from consts.py
        default_colors = [
            Colors.defaults.dust_red,
            Colors.defaults.vermilion,
            Colors.defaults.amber,
            Colors.defaults.blue,
            Colors.defaults.green,
        ]
        
        plot_colors = default_colors[:len(galaxy_data_list)] if len(galaxy_data_list) <= len(default_colors) else None
        
        num_props = len(plot_props)
        fig, axes = plt.subplots(1, num_props)
        fig.set_facecolor("#FFF")
        
        if num_props == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, ax in enumerate(axes):
            cls.compare_total(
                ax, target, plot_props[i], galaxy_data_list, colors=plot_colors,
                wcs_header=wcs_header, fov_vertices=fov_vertices
            )
            
            ptype = TypeParser.parse_type(plot_props[i], unit_dict=galaxy_data_list[0].units)
            
            if ptype == "age":
                ax.set_ylim(9.1, 10.5)
            elif ptype == "metallicity":
                ax.set_ylim(-0.4, 0.4)
            elif ptype == "sfr":
                if ax.containers and ax.containers[0].lines: ax.containers[0].lines[0].set_label("CIGALE")
                ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+2)
                
            # Draw the legend using the existing handle order.
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc=0)
                
            ax.tick_params(axis="both", which="both", direction="in")
            ax.tick_params(axis="both", which="major", width=1.2)
            ax.set_xlim(-0.5, len(galaxy_data_list) - 0.5)
            
            if ptype == "age":
                ylabel_str = r"log$_{10}$[Age/yr]"
            elif ptype == "metallicity":
                ylabel_str = r"log$_{10}[Z/Z_{\odot}]$"
            elif ptype == "sfr":
                ylabel_str = r"SFR [$M_{\odot}/\mathrm{yr}$]"
            else:
                ylabel_str = PlotConfig.ylabels.get(ptype, "Y")
                
            ax.set_ylabel(ylabel_str)
            ax.grid(visible=True, which="major", axis="both", color="#555", alpha=0.3, linewidth=0.5, linestyle="--")

        fig.tight_layout()
        plt.savefig(out_path, format="pdf")
