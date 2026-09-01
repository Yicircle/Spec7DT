import matplotlib.pyplot as plt
import matplotlib.axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import warnings
from astropy.visualization import (
    AsinhStretch, AsymmetricPercentileInterval,
    ImageNormalize,
)
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from scipy.stats import percentileofscore

# from ..utils.file_handler import GalaxyImageSet
from ..utils.utility import useful_functions
from ..utils.file_generator import (
    add_total_flux_rows,
    catalog_row_galaxies,
    galaxy_filter_columns,
)


plt.rcParams["font.family"] = "FreeSerif"

plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


class DrawGalaxy:
    def __init__(self):
        pass

    @staticmethod
    def _is_wildcard(value):
        return value is None or value == "*"

    @classmethod
    def select_images(cls, image_set, galaxy=None, obs=None, band=None):
        """Resolve hierarchical plot selectors to image-set key triples."""
        data = image_set.data
        if cls._is_wildcard(galaxy):
            if not cls._is_wildcard(obs) and not cls._is_wildcard(band):
                selected = [
                    (galaxy_name, obs, band)
                    for galaxy_name in sorted(data)
                    if obs in data[galaxy_name]
                    and band in data[galaxy_name][obs]
                ]
            else:
                selected = [
                    (galaxy_name, observatory, band_name)
                    for galaxy_name in sorted(data)
                    for observatory in sorted(data[galaxy_name])
                    for band_name in sorted(data[galaxy_name][observatory])
                ]
        else:
            if galaxy not in data:
                available = ", ".join(sorted(data)) or "none"
                raise ValueError(
                    f"Unknown galaxy selector '{galaxy}'. Available galaxies: {available}."
                )

            if cls._is_wildcard(obs):
                selected = [
                    (galaxy, observatory, band_name)
                    for observatory in sorted(data[galaxy])
                    for band_name in sorted(data[galaxy][observatory])
                ]
            else:
                if obs not in data[galaxy]:
                    available = ", ".join(sorted(data[galaxy])) or "none"
                    raise ValueError(
                        f"Unknown observatory selector '{obs}' for galaxy '{galaxy}'. "
                        f"Available observatories: {available}."
                    )

                if cls._is_wildcard(band):
                    selected = [
                        (galaxy, obs, band_name)
                        for band_name in sorted(data[galaxy][obs])
                    ]
                else:
                    if band not in data[galaxy][obs]:
                        available = ", ".join(sorted(data[galaxy][obs])) or "none"
                        raise ValueError(
                            f"Unknown band selector '{band}' for {galaxy}/{obs}. "
                            f"Available bands: {available}."
                        )
                    selected = [(galaxy, obs, band)]

        if not selected:
            raise ValueError("No images match the plot_step selectors.")
        return selected

    @classmethod
    def plot_step(cls, image_set, galaxy=None, obs=None, band=None, step=None, show=True):
        selected = cls.select_images(image_set, galaxy=galaxy, obs=obs, band=band)
        exact_selection = not any(
            cls._is_wildcard(value) for value in (galaxy, obs, band)
        )
        if exact_selection:
            return cls.single_galaxy(
                image_set,
                selected[0][0],
                selected[0][1],
                selected[0][2],
                step=step,
                show=show,
            )
        return cls._plot_selected_images(image_set, selected, step=step, show=show)
    
    @classmethod
    def plot_galaxies(cls, image_set, galaxy: str, step: str, show=True):
        selected = cls.select_images(image_set, galaxy=galaxy, obs=None, band=None)
        return cls._plot_selected_images(image_set, selected, step=step, show=show)

    @classmethod
    def _plot_selected_images(cls, image_set, selected, step=None, show=True):
        m, n = useful_functions.find_rec(len(selected))
        fig, axes = plt.subplots(m, n, dpi=160, figsize=(n * 3.6, m * 3.0))
        flat_axes = np.atleast_1d(axes).ravel()

        for ax, (galaxy, obs, band) in zip(flat_axes, selected):
            im_data = image_set.data[galaxy][obs][band]
            norm = ImageNormalize(
                im_data,
                interval=AsymmetricPercentileInterval(50., 99.8),
                stretch=AsinhStretch(),
            )

            im = ax.imshow(im_data, cmap='gray', origin="lower", norm=norm)
            ax.tick_params(axis="both", which="both", direction="in", labelsize=9)
            ax.tick_params(axis="both", which="major", width=1.2)
            ax.set_title(f"{galaxy} / {obs} / {band}", fontsize=11)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            colorbar = fig.colorbar(im, cax=cax, orientation='vertical')
            colorbar.ax.tick_params(labelsize=8)

        for ax in flat_axes[len(selected):]:
            ax.remove()

        if step is not None:
            fig.suptitle(f"Step Name: {step}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94) if step is not None else None)
        if show:
            plt.show()

        return fig, flat_axes[:len(selected)]

    @classmethod
    def plot_sed(
        cls,
        image_set,
        catalog_frame,
        galaxy="*",
        *,
        row="total",
        galaxy_metadata=None,
        metadata_config=None,
        invalid_values=(99.0, -99.0),
    ):
        """Plot total or single-pixel SEDs from a stored pipeline catalog."""
        if not isinstance(row, str):
            raise TypeError("row must be 'total' or an exact '<galaxy>_<pixel>' ID")

        source_df = catalog_frame.data
        row_galaxies = catalog_row_galaxies(source_df)
        available_catalog_galaxies = list(dict.fromkeys(row_galaxies.tolist()))
        wildcard = cls._is_wildcard(galaxy)

        if row.lower() == "total":
            plot_df = add_total_flux_rows(
                source_df,
                galaxy_metadata,
                metadata_config=metadata_config,
                invalid_values=invalid_values,
            )
            selected_galaxies = (
                sorted(available_catalog_galaxies)
                if wildcard
                else [galaxy]
            )
            row_ids = {name: f"{name}_Total" for name in selected_galaxies}
            plot_title = "Integrated SED"
        else:
            matches = source_df.index[source_df["id"].astype(str) == row].tolist()
            if len(matches) != 1:
                raise ValueError(
                    f"Catalog row '{row}' was not found exactly once; found {len(matches)}."
                )
            row_galaxy = catalog_row_galaxies(source_df.loc[matches])
            selected_galaxy = row_galaxy.iloc[0]
            if not wildcard and galaxy != selected_galaxy:
                raise ValueError(
                    f"Row '{row}' belongs to '{selected_galaxy}', not '{galaxy}'."
                )
            plot_df = source_df.copy(deep=True)
            selected_galaxies = [selected_galaxy]
            row_ids = {selected_galaxy: row}
            plot_title = f"Pixel SED: {row}"

        unknown = sorted(set(selected_galaxies) - set(image_set.data))
        absent = sorted(set(selected_galaxies) - set(available_catalog_galaxies))
        if unknown or absent:
            available = ", ".join(sorted(set(image_set.data) & set(available_catalog_galaxies))) or "none"
            requested = ", ".join(unknown + absent)
            raise ValueError(
                f"Unknown galaxy selector(s): {requested}. Available galaxies: {available}."
            )

        m, n = useful_functions.find_rec(len(selected_galaxies))
        fig, axes = plt.subplots(m, n, dpi=160, figsize=(n * 4.2, m * 3.6))
        flat_axes = np.atleast_1d(axes).ravel()
        invalid_numbers = np.asarray(tuple(invalid_values), dtype=float)
        plotted_points = 0

        for ax, galaxy_name in zip(flat_axes, selected_galaxies):
            selected_row = plot_df.loc[
                plot_df["id"].astype(str) == row_ids[galaxy_name]
            ]
            if len(selected_row) != 1:
                raise ValueError(
                    f"Catalog row '{row_ids[galaxy_name]}' was not found exactly once."
                )
            values = selected_row.iloc[0]
            filter_specs = galaxy_filter_columns(
                image_set,
                galaxy_name,
                catalog_frame.catalog_type,
            )
            by_observatory = {}
            skipped_filters = []

            for observatory, band, flux_column, error_column in filter_specs:
                if flux_column not in plot_df.columns or error_column not in plot_df.columns:
                    raise ValueError(
                        f"Catalog is missing {flux_column!r} or {error_column!r} "
                        f"for {galaxy_name}/{observatory}/{band}."
                    )
                try:
                    curve = image_set.filter_inst.get_filter(
                        name=band,
                        facility=observatory,
                    )
                    wavelength = float(curve.pivot_wavelength) / 1.0e4
                    flux = float(values[flux_column])
                    error = float(values[error_column])
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    skipped_filters.append(f"{observatory}/{band}")
                    continue

                invalid_flux = (
                    not np.isfinite(wavelength)
                    or wavelength <= 0
                    or not np.isfinite(flux)
                    or flux <= 0
                    or (invalid_numbers.size and np.isin(flux, invalid_numbers))
                )
                if invalid_flux:
                    continue
                valid_error = (
                    np.isfinite(error)
                    and error >= 0
                    and not (invalid_numbers.size and np.isin(error, invalid_numbers))
                )
                by_observatory.setdefault(observatory, []).append(
                    (wavelength, flux, error if valid_error else np.nan)
                )

            if skipped_filters:
                warnings.warn(
                    f"Skipped filters without usable wavelength/value metadata for "
                    f"{galaxy_name}: {', '.join(skipped_filters)}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            galaxy_points = 0
            for observatory, points in sorted(by_observatory.items()):
                points.sort(key=lambda item: item[0])
                wavelengths = np.asarray([item[0] for item in points])
                fluxes = np.asarray([item[1] for item in points])
                errors = np.asarray([item[2] for item in points])
                yerr = errors if np.any(np.isfinite(errors)) else None
                ax.errorbar(
                    wavelengths,
                    fluxes,
                    yerr=yerr,
                    fmt="o-",
                    markersize=3.5,
                    linewidth=1.0,
                    capsize=2,
                    label=observatory,
                )
                galaxy_points += len(points)

            plotted_points += galaxy_points
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Wavelength [$\mu$m]", fontsize=10)
            flux_unit = catalog_frame.units.get("flux", "mJy")
            ax.set_ylabel(f"Flux density [{flux_unit}]", fontsize=10)
            ax.set_title(galaxy_name, fontsize=11)
            ax.tick_params(axis="both", which="both", labelsize=9, direction="in")
            ax.grid(alpha=0.2, which="both")
            if galaxy_points:
                ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid SED points",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )

        for ax in flat_axes[len(selected_galaxies):]:
            ax.remove()
        if plotted_points == 0:
            plt.close(fig)
            raise ValueError("No valid positive SED fluxes with known filter wavelengths.")

        fig.suptitle(plot_title, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        plt.show()
        return fig, flat_axes[:len(selected_galaxies)]

    @classmethod
    def single_galaxy(cls, image_set, galaxy: str, obs: str, band: str,
                      compass=True, scale=True, step=None, show=True):
        im_data = image_set.data[galaxy][obs][band]
        im_hdr = image_set.header[galaxy][obs][band]
        y, x = im_data.shape
        
        fig, ax = plt.subplots(1, 1, dpi=120, figsize=(9, 9), subplot_kw=dict(projection=WCS(im_hdr)))
            
        _, median, _ = sigma_clipped_stats(im_data, sigma=10.0, mask=np.where(im_data == 0, True, False))
        bg_percent = percentileofscore(np.nan_to_num(im_data.flatten()), median, kind="mean")
        
        norm = ImageNormalize(im_data, interval=AsymmetricPercentileInterval(bg_percent, 99.8), stretch=AsinhStretch())
        
        im = ax.imshow(im_data, cmap='gray', origin="lower", norm=norm)
        
        if compass:
            useful_functions.plot_compass_rose(ax, x * 0.9, y * 0.1, WCS(im_hdr), size=1/12*x, color='white')
            
        if scale:
            useful_functions.plot_scale(ax, x * 0.1, y * 0.1, WCS(im_hdr), size=120, color='white')
        
        ax.tick_params(axis="both", which="both", direction="in", color="#CCC", labelsize=9)
        ax.tick_params(axis="both", which="major", width=1.2)
        ax.set_xlabel(r"$\alpha_{2000}$", fontsize=11)
        ax.set_ylabel(r"$\delta_{2000}$", fontsize=11)
        ax.set_title(f"{galaxy} / {obs} / {band}", fontsize=12)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=matplotlib.axes.Axes)
        cax.tick_params(axis="y", which="both", direction="in", color="#000", labelsize=9)
        colorbar = fig.colorbar(im, cax=cax, orientation='vertical')
        colorbar.set_label("mJy", fontsize=11)
        
        if step is not None:
            fig.suptitle(f"Step Name: {step}", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.94) if step is not None else None)
        
        if show:
            plt.show()
        
        return fig, ax
