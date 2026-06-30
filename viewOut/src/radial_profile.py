from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from photutils.aperture import CircularAperture, CircularAnnulus


@dataclass
class RadialBinAudit:
    index: int
    a_in: float
    a_out: float
    b_in: float
    b_out: float
    radius_arcsec: float
    n_masked: int
    n_valid: int
    valid_frac: float
    sum_weights: float
    status: str
    skip_reason: str = None
    warnings: list = field(default_factory=list)
    mean: float = np.nan
    std: float = np.nan
    mask: np.ndarray = None


@dataclass
class RadialProfileAudit:
    image_shape: tuple
    center: tuple
    max_radius: float
    platescale: float
    n_steps: int
    min_valid_frac: float
    axis_ratio: float
    theta: float
    a_edges: np.ndarray
    bins: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _as_2d_float_array(values, name, shape=None):
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match image shape {shape}.")
    return array


def _validate_radial_inputs(
    image,
    center,
    max_radius,
    platescale,
    error_image,
    weight,
    weight_err,
    n_steps,
    min_valid_frac,
    axis_ratio,
):
    image = _as_2d_float_array(image, "image")
    shape = image.shape

    error_image = _as_2d_float_array(error_image, "error_image", shape)
    weight = _as_2d_float_array(weight, "weight", shape)
    weight_err = _as_2d_float_array(weight_err, "weight_err", shape)

    if not np.isfinite(platescale) or platescale <= 0:
        raise ValueError("platescale must be a finite positive value.")
    if not np.isfinite(max_radius) or max_radius <= 0:
        raise ValueError("max_radius must be a finite positive value.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if not np.isfinite(axis_ratio) or not (0 < axis_ratio <= 1):
        raise ValueError("axis_ratio must satisfy 0 < axis_ratio <= 1.")
    if not np.isfinite(min_valid_frac) or not (0 <= min_valid_frac <= 1):
        raise ValueError("min_valid_frac must satisfy 0 <= min_valid_frac <= 1.")

    center_array = np.asarray(center, dtype=float)
    if center_array.shape != (2,) or not np.all(np.isfinite(center_array)):
        raise ValueError("center must contain two finite values.")
    center = tuple(center_array)
    x_center, y_center = center
    if not (0 <= x_center < shape[1] and 0 <= y_center < shape[0]):
        raise ValueError("center must be within the image bounds.")

    if weight is None:
        weight = np.ones_like(image, dtype=float)
    if weight_err is None:
        weight_err = np.zeros_like(image, dtype=float)

    return image, center, error_image, weight, weight_err


def elliptical_annulus_mask(positions, a_in, a_out, b_in, b_out, theta, shape):
    """
    Generate a boolean mask for an elliptical annulus.

    Parameters
    ----------
    positions : tuple
        (x, y) center position of the annulus
    a_in : float
        Inner semi-major axis
    a_out : float
        Outer semi-major axis
    b_in : float
        Inner semi-minor axis
    b_out : float
        Outer semi-minor axis
    theta : float
        Rotation angle in degrees (counter-clockwise from x-axis)
    shape : tuple
        (height, width) shape of the output mask

    Returns
    -------
    mask : ndarray
        Boolean mask where True indicates pixels inside the annulus
    """
    if a_out <= 0 or b_out <= 0:
        return np.zeros(shape, dtype=bool)

    y, x = np.ogrid[:shape[0], :shape[1]]
    x_c, y_c = positions

    theta_rad = np.deg2rad(180 - theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    dx = x - x_c
    dy = y - y_c

    x_rot = cos_theta * dx + sin_theta * dy
    y_rot = -sin_theta * dx + cos_theta * dy

    outer_ellipse = (x_rot / a_out) ** 2 + (y_rot / b_out) ** 2
    if a_in <= 0 or b_in <= 0:
        return outer_ellipse <= 1

    inner_ellipse = (x_rot / a_in) ** 2 + (y_rot / b_in) ** 2
    return (outer_ellipse <= 1) & (inner_ellipse >= 1)


def _circular_mask(center, a_in, a_out, shape):
    if a_out <= 0:
        return np.zeros(shape, dtype=bool)
    if a_in <= 0:
        aperture = CircularAperture(positions=[center], r=a_out)
    else:
        aperture = CircularAnnulus(positions=[center], r_in=a_in, r_out=a_out)
    mask_image = aperture.to_mask(method="exact")[0].to_image(shape)
    if mask_image is None:
        return np.zeros(shape, dtype=bool)
    return mask_image > 0


def _make_annulus_mask(center, a_in, a_out, b_in, b_out, theta, axis_ratio, shape):
    if axis_ratio == 1.0:
        return _circular_mask(center, a_in, a_out, shape)
    return elliptical_annulus_mask(
        positions=center,
        a_in=a_in,
        a_out=a_out,
        b_in=b_in,
        b_out=b_out,
        theta=theta,
        shape=shape,
    )


def _bin_warnings(i, n_steps, a_in, a_out, b_in, max_radius, platescale, mask, image_shape):
    warnings = []
    if a_in <= 0 or b_in <= 0:
        warnings.append("inner radius is non-positive; using a filled aperture for this bin")
    if i == n_steps - 1:
        warnings.append("legacy final bin uses a synthetic outer edge because no next edge exists")
    if a_out * platescale > max_radius:
        warnings.append("outer edge exceeds max_radius because legacy binning extends by one pixel")
    if mask.shape != tuple(image_shape):
        warnings.append("mask shape does not match image shape")
    if mask.dtype != bool:
        warnings.append("mask dtype is not bool")
    return warnings


def _append_audit_bin(
    audit,
    store_mask,
    index,
    a_in,
    a_out,
    b_in,
    b_out,
    radius_arcsec,
    mask,
    valid,
    sum_weights,
    status,
    skip_reason=None,
    warnings=None,
    mean=np.nan,
    std=np.nan,
):
    if audit is None:
        return

    n_masked = int(mask.sum())
    n_valid = int(valid.sum())
    valid_frac = float(n_valid / n_masked) if n_masked > 0 else 0.0
    audit.bins.append(
        RadialBinAudit(
            index=index,
            a_in=float(a_in),
            a_out=float(a_out),
            b_in=float(b_in),
            b_out=float(b_out),
            radius_arcsec=float(radius_arcsec),
            n_masked=n_masked,
            n_valid=n_valid,
            valid_frac=valid_frac,
            sum_weights=float(sum_weights) if np.isfinite(sum_weights) else np.nan,
            status=status,
            skip_reason=skip_reason,
            warnings=list(warnings or []),
            mean=float(mean) if np.isfinite(mean) else np.nan,
            std=float(std) if np.isfinite(std) else np.nan,
            mask=mask.copy() if store_mask else None,
        )
    )


def _make_profile_audit(image_shape, center, max_radius, platescale, n_steps, min_valid_frac, axis_ratio, theta, a_edges):
    warnings = [
        "legacy radial binning uses n_steps edge values for n_steps bins",
        "legacy radial binning extends the nominal outer edge by one pixel",
    ]
    if a_edges[-1] * platescale > max_radius:
        warnings.append(
            f"last configured edge {a_edges[-1] * platescale:.6g} arcsec exceeds max_radius {max_radius:.6g}"
        )
    return RadialProfileAudit(
        image_shape=tuple(image_shape),
        center=tuple(center),
        max_radius=float(max_radius),
        platescale=float(platescale),
        n_steps=int(n_steps),
        min_valid_frac=float(min_valid_frac),
        axis_ratio=float(axis_ratio),
        theta=float(theta),
        a_edges=np.asarray(a_edges, dtype=float).copy(),
        warnings=warnings,
    )


def radial_property(
    image,
    center,
    max_radius,
    platescale,
    error_image=None,
    weight=None,
    weight_err=None,
    n_steps=50,
    min_valid_frac=2 / 3,
    axis_ratio=0.55,
    theta=90.0,
    return_audit=False,
    audit_pdf_path=None,
    store_audit_masks=None,
):
    """
    Calculate radial properties (weighted mean, dispersion, propagated error)
    within elliptical annuli.
    """
    image, center, error_image, weight, weight_err = _validate_radial_inputs(
        image,
        center,
        max_radius,
        platescale,
        error_image,
        weight,
        weight_err,
        n_steps,
        min_valid_frac,
        axis_ratio,
    )

    if store_audit_masks is None:
        store_audit_masks = audit_pdf_path is not None
    if audit_pdf_path is not None:
        store_audit_masks = True
    audit_enabled = return_audit or audit_pdf_path is not None

    max_r_pix = max_radius / platescale
    a_edges = np.linspace(0, max_r_pix + 1, n_steps)

    audit = None
    if audit_enabled:
        audit = _make_profile_audit(
            image.shape,
            center,
            max_radius,
            platescale,
            n_steps,
            min_valid_frac,
            axis_ratio,
            theta,
            a_edges,
        )

    radii_arcsec = []
    means = []
    stds = []

    for i in range(n_steps):
        a_in = a_edges[i]
        a_out = a_edges[i + 1] if a_in != a_edges[-1] else a_edges[i] + 0.1
        b_in = a_in * axis_ratio
        b_out = a_out * axis_ratio
        mid_a = 0.5 * (a_in + a_out)
        radius_arcsec = mid_a * platescale

        raw_mask = _make_annulus_mask(center, a_in, a_out, b_in, b_out, theta, axis_ratio, image.shape)
        warnings = _bin_warnings(i, n_steps, a_in, a_out, b_in, max_radius, platescale, raw_mask, image.shape)
        mask = np.asarray(raw_mask, dtype=bool)
        vals = image[mask]
        valid = np.isfinite(vals) & (vals != 0.0)
        n_total = int(mask.sum())
        n_valid = int(valid.sum())

        vweights = weight[mask][valid] if n_total > 0 else np.array([], dtype=float)
        sum_weights = np.nansum(vweights) if vweights.size > 0 else 0.0

        if n_total == 0:
            _append_audit_bin(
                audit,
                store_audit_masks,
                i,
                a_in,
                a_out,
                b_in,
                b_out,
                radius_arcsec,
                mask,
                valid,
                sum_weights,
                "skipped_no_mask",
                skip_reason="annulus mask contains no pixels",
                warnings=warnings,
            )
            continue

        if n_valid < min_valid_frac * n_total:
            _append_audit_bin(
                audit,
                store_audit_masks,
                i,
                a_in,
                a_out,
                b_in,
                b_out,
                radius_arcsec,
                mask,
                valid,
                sum_weights,
                "skipped_low_valid_frac",
                skip_reason="valid pixel fraction is below min_valid_frac",
                warnings=warnings,
            )
            continue

        if i == 0:
            row = int(center[1])
            col = int(center[0])
            val = image[row, col]
            mean = np.nan
            std = np.nan
            skip_reason = None
            if np.isfinite(val) and val != 0.0:
                mean = val
                std = 0.0
                means.append(val)
                radii_arcsec.append(0)
                stds.append(0.0)
            else:
                skip_reason = "center pixel is invalid"
            _append_audit_bin(
                audit,
                store_audit_masks,
                i,
                a_in,
                a_out,
                b_in,
                b_out,
                0.0,
                mask,
                valid,
                sum_weights,
                "center",
                skip_reason=skip_reason,
                warnings=warnings,
                mean=mean,
                std=std,
            )
            continue

        if not np.isfinite(sum_weights) or sum_weights <= 0:
            _append_audit_bin(
                audit,
                store_audit_masks,
                i,
                a_in,
                a_out,
                b_in,
                b_out,
                radius_arcsec,
                mask,
                valid,
                sum_weights,
                "skipped_nonpositive_weight",
                skip_reason="sum of valid weights is not finite and positive",
                warnings=warnings,
            )
            continue

        vvals = vals[valid]
        val_mean = np.nansum(vvals * vweights) / sum_weights
        val_std = np.sqrt(np.nansum(vweights * (vvals - val_mean) ** 2) / sum_weights)

        means.append(val_mean)
        stds.append(val_std)
        radii_arcsec.append(radius_arcsec)

        recalculated_mean = np.nansum(vvals * vweights) / sum_weights
        recalculated_std = np.sqrt(np.nansum(vweights * (vvals - recalculated_mean) ** 2) / sum_weights)
        if not np.isclose(val_mean, recalculated_mean, rtol=1e-12, atol=1e-12):
            warnings.append("weighted mean failed recomputation check")
        if not np.isclose(val_std, recalculated_std, rtol=1e-12, atol=1e-12):
            warnings.append("weighted standard deviation failed recomputation check")

        _append_audit_bin(
            audit,
            store_audit_masks,
            i,
            a_in,
            a_out,
            b_in,
            b_out,
            radius_arcsec,
            mask,
            valid,
            sum_weights,
            "used",
            warnings=warnings,
            mean=val_mean,
            std=val_std,
        )

    radii_arcsec = np.array(radii_arcsec)
    means = np.array(means)
    stds = np.array(stds)

    if audit_pdf_path is not None:
        save_radial_profile_check_plots(audit, image, audit_pdf_path)

    if return_audit:
        return radii_arcsec, means, stds, audit
    return radii_arcsec, means, stds


def _finite_limits(values, lower=2, upper=98):
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        return None, None
    return np.nanpercentile(finite_values, lower), np.nanpercentile(finite_values, upper)


def _audit_masks(audit):
    return [record.mask for record in audit.bins if record.mask is not None]


def _plot_annulus_overlay(pdf, audit, image):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    vmin, vmax = _finite_limits(image)
    ax.imshow(image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    for record in audit.bins:
        if record.mask is None or record.n_masked == 0:
            continue
        color = "#FFCC33" if record.status in ("used", "center") else "#FF6666"
        ax.contour(record.mask.astype(float), levels=[0.5], colors=color, linewidths=0.6)
    ax.plot(audit.center[0], audit.center[1], marker="+", color="#FFFFFF", markersize=10)
    ax.set_title("Radial annulus overlay")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_mask_grid(pdf, audit):
    import math
    import matplotlib.pyplot as plt

    records = [record for record in audit.bins if record.mask is not None]
    if not records:
        return

    ncols = min(5, max(1, len(records)))
    nrows = math.ceil(len(records) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows), dpi=160, squeeze=False)

    for ax, record in zip(axes.flat, records):
        ax.imshow(record.mask, origin="lower", cmap="gray_r")
        title = f"bin {record.index}: {record.status}\nN={record.n_masked}, valid={record.n_valid}"
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flat[len(records):]:
        ax.axis("off")

    fig.suptitle("Per-bin masks", fontsize=12)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_profile_page(pdf, audit):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    used = [record for record in audit.bins if record.status in ("used", "center") and np.isfinite(record.mean)]
    skipped = [record for record in audit.bins if record.status.startswith("skipped")]

    if used:
        x = np.array([record.radius_arcsec for record in used])
        y = np.array([record.mean for record in used])
        yerr = np.array([record.std for record in used])
        ax.plot(x, y, color="#3366AA", marker="o", lw=1.2, label="used")
        ax.fill_between(x, y - 0.5 * yerr, y + 0.5 * yerr, color="#3366AA", alpha=0.18)

    if skipped:
        y_marker = np.nanmin([record.mean for record in used]) if used else 0.0
        ax.scatter(
            [record.radius_arcsec for record in skipped],
            np.full(len(skipped), y_marker),
            marker="x",
            color="#CC3333",
            label="skipped bins",
        )

    ax.set_title("Radial profile audit")
    ax.set_xlabel("Radius [arcsec]")
    ax.set_ylabel("Mean value")
    ax.grid(visible=True, alpha=0.3, linestyle="--", linewidth=0.6)
    if used or skipped:
        ax.legend(fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _format_audit_value(value):
    if value is None:
        return ""
    return f"{value:.4g}" if np.isfinite(value) else "nan"


def _plot_summary_table(pdf, audit, rows_per_page=22):
    import matplotlib.pyplot as plt

    columns = [
        "bin",
        "a_in",
        "a_out",
        "r_arcsec",
        "masked",
        "valid",
        "valid_frac",
        "sum_w",
        "status",
        "reason",
        "warnings",
    ]
    rows = []
    for record in audit.bins:
        rows.append(
            [
                str(record.index),
                _format_audit_value(record.a_in),
                _format_audit_value(record.a_out),
                _format_audit_value(record.radius_arcsec),
                str(record.n_masked),
                str(record.n_valid),
                _format_audit_value(record.valid_frac),
                _format_audit_value(record.sum_weights),
                record.status,
                record.skip_reason or "",
                "; ".join(record.warnings),
            ]
        )

    for start in range(0, len(rows), rows_per_page):
        chunk = rows[start:start + rows_per_page]
        fig, ax = plt.subplots(figsize=(16, 0.48 * len(chunk) + 1.8), dpi=160)
        ax.axis("off")
        ax.set_title("Radial annulus audit table", fontsize=14, pad=12)
        table = ax.table(cellText=chunk, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1.0, 1.35)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#EAEAEA")
            elif columns[col] == "status":
                text = cell.get_text().get_text()
                cell.set_facecolor("#DFF0D8" if text in ("used", "center") else "#F2DEDE")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def save_radial_profile_check_plots(audit, image, out_path):
    if audit is None:
        raise ValueError("audit is required to save radial profile check plots.")
    if not _audit_masks(audit):
        raise ValueError("audit masks are required to save radial profile check plots.")

    from matplotlib.backends.backend_pdf import PdfPages

    out_path = Path(out_path)
    with PdfPages(out_path) as pdf:
        _plot_annulus_overlay(pdf, audit, image)
        _plot_mask_grid(pdf, audit)
        _plot_profile_page(pdf, audit)
        _plot_summary_table(pdf, audit)

    return out_path
