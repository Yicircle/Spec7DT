from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

try:
    from .consts import Colors, PlotConfig
    from .utility import TypeParser
except ImportError:
    from consts import Colors, PlotConfig
    from utility import TypeParser


@dataclass
class MaskedPropertyAudit:
    dataset: str
    target: str
    property_name: str
    ptype: str
    n_masked: int
    n_valid: int
    dist_factor: float
    median: float
    std: float
    final_median: float
    final_std: float
    delta_median: float
    delta_std: float
    passed: bool
    values: np.ndarray
    mask: np.ndarray
    pixel_vertices: np.ndarray
    plot_map: np.ndarray


def _as_wcs(wcs_header):
    if isinstance(wcs_header, WCS):
        return wcs_header
    if isinstance(wcs_header, (str, Path)):
        return WCS(fits.getheader(wcs_header))
    return WCS(wcs_header)


def _select_wcs_header(wcs_header, index, g_data, target):
    if isinstance(wcs_header, (list, tuple)):
        return wcs_header[index]
    if isinstance(wcs_header, dict):
        return wcs_header.get(g_data.name, wcs_header.get(target, wcs_header.get("default")))
    return wcs_header


def _looks_like_vertices(value):
    if isinstance(value, SkyCoord):
        return True
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2


def _select_fov_vertices(fov_vertices, index, g_data, target, num_items):
    if isinstance(fov_vertices, dict):
        return fov_vertices.get(g_data.name, fov_vertices.get(target, fov_vertices.get("default")))
    if (
        isinstance(fov_vertices, (list, tuple))
        and len(fov_vertices) == num_items
        and not _looks_like_vertices(fov_vertices)
    ):
        return fov_vertices[index]
    return fov_vertices


def _as_skycoord(fov_vertices):
    if isinstance(fov_vertices, SkyCoord):
        return fov_vertices.icrs
    vertices = np.asarray(fov_vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("fov_vertices must be a SkyCoord or an array-like object with shape (N, 2).")
    return SkyCoord(vertices[:, 0], vertices[:, 1], unit="deg", frame="icrs")


def _pixel_vertices(wcs, fov_vertices):
    coords = _as_skycoord(fov_vertices)
    x_vertices, y_vertices = wcs.world_to_pixel(coords)
    vertices = np.column_stack((x_vertices, y_vertices))
    if not np.all(np.isfinite(vertices)):
        raise ValueError("FOV vertices could not be converted to finite pixel coordinates.")
    return vertices


def _make_fov_mask(shape, pixel_vertices):
    y_grid, x_grid = np.mgrid[:shape[0], :shape[1]]
    pixel_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    fov_path = MplPath(pixel_vertices)
    return fov_path.contains_points(pixel_points, radius=0.5).reshape(shape)


def _dist_factor(g_data, target, ptype):
    if ptype in ["mass", "luminosity", "sfr"]:
        return (g_data.dist[target] / g_data.dist_red[target]) ** 2
    return 1.0


def _valid_pixels(values, ptype):
    valid = np.isfinite(values) & (values != 0.0)
    if ptype in ["age", "metallicity"]:
        valid &= values > 0.0
    return valid


def _to_plot_unit(values, ptype, z_sol):
    if ptype == "age":
        return np.log10(values)
    if ptype == "metallicity":
        return np.log10(values / z_sol)
    return values


def _plot_unit_map(image, ptype, z_sol, dist_factor):
    plot_map = np.full(image.shape, np.nan, dtype=float)
    valid = _valid_pixels(image, ptype)
    plot_map[valid] = _to_plot_unit(image[valid], ptype, z_sol)
    if ptype in ["mass", "luminosity", "sfr"]:
        plot_map *= dist_factor
    return plot_map


def compute_masked_property_audit(
    g_data,
    target,
    property_name,
    wcs_header,
    fov_vertices,
    z_sol=0.02,
    rtol=1e-10,
    atol=1e-12,
):
    ptype = TypeParser.parse_type(property_name, unit_dict=g_data.units)
    image = np.asarray(g_data.get_image(target, property_name), dtype=float)
    wcs = _as_wcs(wcs_header)
    vertices = _pixel_vertices(wcs, fov_vertices)
    mask = _make_fov_mask(image.shape, vertices)
    dist_factor = _dist_factor(g_data, target, ptype)

    masked_values = image[mask]
    valid_values = masked_values[_valid_pixels(masked_values, ptype)]
    if valid_values.size == 0:
        raise ValueError(f"No valid pixels found inside the FOV for {g_data.name}: {property_name}.")

    values = _to_plot_unit(valid_values, ptype, z_sol)
    if ptype in ["mass", "luminosity", "sfr"]:
        values = values * dist_factor

    median = float(np.nanmedian(values))
    std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0

    final_median, final_std = g_data.get_fov_property_stats(
        target, property_name, wcs_header, fov_vertices, ptype=ptype, z_sol=z_sol
    )
    if ptype in ["mass", "luminosity", "sfr"]:
        final_median *= dist_factor
        final_std *= dist_factor

    delta_median = float(median - final_median)
    delta_std = float(std - final_std)
    passed = bool(
        np.isclose(median, final_median, rtol=rtol, atol=atol)
        and np.isclose(std, final_std, rtol=rtol, atol=atol)
    )

    return MaskedPropertyAudit(
        dataset=g_data.name,
        target=target,
        property_name=property_name,
        ptype=ptype,
        n_masked=int(mask.sum()),
        n_valid=int(values.size),
        dist_factor=float(dist_factor),
        median=median,
        std=std,
        final_median=float(final_median),
        final_std=float(final_std),
        delta_median=delta_median,
        delta_std=delta_std,
        passed=passed,
        values=values,
        mask=mask,
        pixel_vertices=vertices,
        plot_map=_plot_unit_map(image, ptype, z_sol, dist_factor),
    )


def compute_global_comparison_audit(
    galaxy_data_list,
    target,
    plot_props,
    wcs_header,
    fov_vertices,
    z_sol=0.02,
    rtol=1e-10,
    atol=1e-12,
):
    records = []
    num_items = len(galaxy_data_list)
    for index, g_data in enumerate(galaxy_data_list):
        selected_wcs_header = _select_wcs_header(wcs_header, index, g_data, target)
        selected_vertices = _select_fov_vertices(fov_vertices, index, g_data, target, num_items)
        for property_name in plot_props:
            records.append(
                compute_masked_property_audit(
                    g_data,
                    target,
                    property_name,
                    selected_wcs_header,
                    selected_vertices,
                    z_sol=z_sol,
                    rtol=rtol,
                    atol=atol,
                )
            )
    return records


def _records_for_property(records, property_name):
    return [record for record in records if record.property_name == property_name]


def _finite_limits(arrays, lower=2, upper=98):
    finite_arrays = [array[np.isfinite(array)] for array in arrays if np.any(np.isfinite(array))]
    if not finite_arrays:
        return None, None
    values = np.concatenate(finite_arrays)
    if values.size == 0:
        return None, None
    return np.nanpercentile(values, lower), np.nanpercentile(values, upper)


def _format_value(value):
    return f"{value:.4g}" if np.isfinite(value) else "nan"


def _make_grid(num_items, max_cols=3, figsize_per_panel=(4.0, 3.6)):
    ncols = min(max_cols, max(1, num_items))
    nrows = math.ceil(num_items / ncols)
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    return nrows, ncols, figsize


def _draw_polygon(ax, pixel_vertices, color="#FFFFFF", lw=1.2):
    patch = Polygon(pixel_vertices, closed=True, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(patch)


def _plot_geometry_page(pdf, records, galaxy_data_list, reference_property):
    ref_records = _records_for_property(records, reference_property)
    nrows, ncols, figsize = _make_grid(len(ref_records))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=160, squeeze=False)
    fig.suptitle(f"FOV geometry check: {reference_property}", fontsize=14)

    vmin, vmax = _finite_limits([record.plot_map for record in ref_records])
    for ax, record in zip(axes.flat, ref_records):
        im = ax.imshow(record.plot_map, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        _draw_polygon(ax, record.pixel_vertices, color="#FFFFFF", lw=1.4)
        ax.contour(record.mask.astype(float), levels=[0.5], colors="#FFCC33", linewidths=1.0)
        ax.set_title(record.dataset, fontsize=10)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        ax.text(
            0.03,
            0.97,
            f"masked={record.n_masked}\nvalid={record.n_valid}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#FFFFFF",
            bbox=dict(facecolor="#000000", alpha=0.55, edgecolor="none", pad=3),
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.flat[len(ref_records):]:
        ax.axis("off")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_masked_map_pages(pdf, records, plot_props):
    for property_name in plot_props:
        prop_records = _records_for_property(records, property_name)
        nrows, ncols, figsize = _make_grid(len(prop_records))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=160, squeeze=False)
        fig.suptitle(f"Masked map check: {property_name}", fontsize=14)

        masked_maps = [np.where(record.mask, record.plot_map, np.nan) for record in prop_records]
        vmin, vmax = _finite_limits(masked_maps)

        for ax, record, masked_map in zip(axes.flat, prop_records, masked_maps):
            im = ax.imshow(masked_map, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            _draw_polygon(ax, record.pixel_vertices, color="#FFFFFF", lw=1.2)
            ax.contour(record.mask.astype(float), levels=[0.5], colors="#FFCC33", linewidths=0.9)
            ax.set_title(record.dataset, fontsize=10)
            ax.set_xlabel("x [pix]")
            ax.set_ylabel("y [pix]")
            ax.text(
                0.03,
                0.97,
                f"median={_format_value(record.median)}\nstd={_format_value(record.std)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="#FFFFFF",
                bbox=dict(facecolor="#000000", alpha=0.55, edgecolor="none", pad=3),
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes.flat[len(prop_records):]:
            ax.axis("off")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _plot_histogram_pages(pdf, records, plot_props):
    for property_name in plot_props:
        prop_records = _records_for_property(records, property_name)
        nrows, ncols, figsize = _make_grid(len(prop_records))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=160, squeeze=False)
        fig.suptitle(f"Masked pixel distribution: {property_name}", fontsize=14)

        for ax, record in zip(axes.flat, prop_records):
            ax.hist(record.values, bins=35, color=Colors.defaults.blue, alpha=0.75)
            ax.axvline(record.median, color=Colors.defaults.vermilion, lw=1.8, label="median")
            ax.axvspan(
                record.median - record.std,
                record.median + record.std,
                color=Colors.defaults.amber,
                alpha=0.25,
                label="median +/- std",
            )
            ax.set_title(record.dataset, fontsize=10)
            ax.set_xlabel("value in final plot unit")
            ax.set_ylabel("N pixels")
            ax.text(
                0.97,
                0.97,
                f"N={record.n_valid}\nmedian={_format_value(record.median)}\nstd={_format_value(record.std)}",
                transform=ax.transAxes,
                va="top",
                ha="right",
                fontsize=8,
                bbox=dict(facecolor="#FFFFFF", alpha=0.85, edgecolor="none", pad=3),
            )
            ax.legend(fontsize=8)

        for ax in axes.flat[len(prop_records):]:
            ax.axis("off")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _draw_reference_values(ax, ptype, base_data, target):
    refs = PlotConfig.global_refs.get(ptype, [])
    for ref in refs:
        val = ref.get("value")
        if val is None and "value_calc" in ref:
            val = ref["value_calc"](base_data.dist[target])

        color = ref.get("color", "#333")
        ls = ref.get("ls", "-")
        label = ref.get("label", "")
        xmin = ref.get("xmin", 0.0)
        xmax = ref.get("xmax", 1.0)

        if "span" in ref and ref["span"]:
            span_lower, span_upper = ref["span"]
            ax.axhspan(
                span_lower,
                span_upper,
                xmin=xmin,
                xmax=xmax,
                alpha=0.15,
                color=color,
                label=label if val is None else None,
            )

        if val is not None:
            ax.axhline(val, xmin=xmin, xmax=xmax, ls=ls, lw=1.5, color=color, label=label)


def _ylabel_for_ptype(ptype):
    if ptype == "age":
        return r"log$_{10}$[Age/yr]"
    if ptype == "metallicity":
        return r"log$_{10}[Z/Z_{\odot}]$"
    if ptype == "sfr":
        return r"SFR [$M_{\odot}/\mathrm{yr}$]"
    return PlotConfig.ylabels.get(ptype, "Y")


def _plot_global_audit_page(pdf, records, galaxy_data_list, target, plot_props):
    num_props = len(plot_props)
    fig, axes = plt.subplots(1, num_props, dpi=160, figsize=(4.2 * num_props, 3.8), squeeze=False)
    fig.suptitle("Global comparison audit", fontsize=14)
    colors = [
        Colors.defaults.vermilion,
        Colors.defaults.amber,
        Colors.defaults.blue,
        Colors.defaults.green,
        Colors.defaults.dust_red,
    ]
    if len(galaxy_data_list) > len(colors):
        colors = plt.cm.tab10(np.linspace(0, 1, len(galaxy_data_list)))
    else:
        colors = colors[:len(galaxy_data_list)]

    for ax, property_name in zip(axes.flat, plot_props):
        prop_records = _records_for_property(records, property_name)
        ptype = prop_records[0].ptype
        x_values = np.arange(len(prop_records))
        y_values = np.array([record.median for record in prop_records])
        y_errs = np.array([record.std for record in prop_records])
        labels = [record.dataset for record in prop_records]

        _draw_reference_values(ax, ptype, galaxy_data_list[0], target)
        ax.scatter(x_values, y_values, c=colors, marker=".", s=50, zorder=3)
        for x_value, y_value, y_err, color, record in zip(x_values, y_values, y_errs, colors, prop_records):
            ax.errorbar(x_value, y_value, y_err, color=color, alpha=1, capsize=8, zorder=2)
            status = "OK" if record.passed else "CHECK"
            ax.annotate(
                f"{status}\nΔy={record.delta_median:.1e}\nΔe={record.delta_std:.1e}",
                (x_value, y_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_xticks(x_values, labels=labels)
        ax.set_xlim(-0.5, len(prop_records) - 0.5)
        ax.set_ylabel(_ylabel_for_ptype(ptype))
        ax.set_title(property_name, fontsize=10)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=10)
        ax.tick_params(axis="both", which="major", width=1.2)
        ax.grid(visible=True, which="major", axis="both", color="#555", alpha=0.3, linewidth=0.5, linestyle="--")

        if ptype == "age":
            ax.set_ylim(9.1, 10.5)
        elif ptype == "metallicity":
            ax.set_ylim(-0.4, 0.4)
        elif ptype == "sfr":
            ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 2)

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=0, fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _table_rows(records):
    rows = []
    for record in records:
        rows.append(
            [
                record.dataset,
                record.property_name,
                str(record.n_masked),
                str(record.n_valid),
                _format_value(record.median),
                _format_value(record.std),
                _format_value(record.final_median),
                _format_value(record.final_std),
                f"{record.delta_median:.1e}",
                f"{record.delta_std:.1e}",
                "PASS" if record.passed else "FAIL",
            ]
        )
    return rows


def _plot_table_pages(pdf, records, rows_per_page=18):
    columns = [
        "dataset",
        "property",
        "masked",
        "valid",
        "median",
        "std",
        "final median",
        "final std",
        "delta y",
        "delta err",
        "status",
    ]
    rows = _table_rows(records)
    for start in range(0, len(rows), rows_per_page):
        chunk = rows[start:start + rows_per_page]
        fig, ax = plt.subplots(figsize=(15, 0.55 * len(chunk) + 1.8), dpi=160)
        ax.axis("off")
        ax.set_title("Numerical consistency table", fontsize=14, pad=12)
        table = ax.table(cellText=chunk, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.35)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#EAEAEA")
            elif columns[col] == "status":
                cell.set_facecolor("#DFF0D8" if cell.get_text().get_text() == "PASS" else "#F2DEDE")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def save_global_comparison_check_plots(
    galaxy_data_list,
    target,
    plot_props,
    wcs_header,
    fov_vertices,
    out_path="global_comparison_check.pdf",
    reference_property=None,
    z_sol=0.02,
    rtol=1e-10,
    atol=1e-12,
):
    records = compute_global_comparison_audit(
        galaxy_data_list,
        target,
        plot_props,
        wcs_header,
        fov_vertices,
        z_sol=z_sol,
        rtol=rtol,
        atol=atol,
    )

    reference_property = reference_property or plot_props[0]
    out_path = Path(out_path)
    with PdfPages(out_path) as pdf:
        _plot_geometry_page(pdf, records, galaxy_data_list, reference_property)
        _plot_masked_map_pages(pdf, records, plot_props)
        _plot_histogram_pages(pdf, records, plot_props)
        _plot_global_audit_page(pdf, records, galaxy_data_list, target, plot_props)
        _plot_table_pages(pdf, records)

    return records
