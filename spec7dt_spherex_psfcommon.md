# Change request: import PSF-homogenized SPHEREx mosaic cubes into Spec7DT

Audience: the coding agent working on Spec7DT (v0.15.0, source at
`/home/wohylee04/anaconda3/envs/spec7dt/lib/python3.12/site-packages/Spec7DT`).
Producer of the cubes: `/data9/wohylee/4_Python_codes/SPHEREx_Query` (read-only for you).

Follow Spec7DT's `AGENT.md` working rules: **propose the plan and diff to the user
before editing**, keep `PSF_safe.py` and `src/Spec7DT/psf/` instrument-independent,
and keep checkpoint manifests backward-compatible.

## 1. Goal and scope

`SpherexMosaicImporter` (`utils/spherex_cube.py`) must also load the new
**PSF-homogenized** SPHEREx mosaic cubes correctly, with the smallest possible
change. Requirements:

* The behaviour for native (non-homogenized) mosaics must not change. Import
  results must be bit-identical (regression test T1).
* All SPHEREx-specific logic stays in `utils/spherex_cube.py` and the SPHEREx
  handlers (`handlers/spherex_psf.py`, `handlers/spherex_mosaic_psf.py`, or a new
  `handlers/spherex_psfcommon.py`).
* Generic pipeline steps are not modified by the SPHEREx part (§4). §5 lists the
  small generic changes O1–O4, which the user has approved (§8).

Non-goals: rebuilding PSFs, re-registering WCS, or re-estimating errors of
these cubes. All three were done and validated by the producer.

## 2. What the new product is (verified facts)

Example files (all exist):

| Role | Path |
|---|---|
| Homogenized, QR2, full coverage | `/data9/wohylee/4_Python_codes/SPHEREx_Query/spherex_output/NGC_3627/psf_v3/NGC3627-SPHEREx-MOSAIC-MOSAIC-AREA-PSFCOMMON.fits` |
| Homogenized, QR3, empty planes | `/data9/wohylee/4_Python_codes/SPHEREx_Query/spherex_output/NGC_0628_qr3/psf_v1/NGC0628-SPHEREx-MOSAIC-MOSAIC-AREA-PSFCOMMON.fits` |
| Native mosaic (regression) | `/data9/wohylee/4_Python_codes/SPHEREx_Query/spherex_output/NGC_0628_qr3/NGC0628-SPHEREx-MOSAIC.fits` |

**Detection.** In the primary header, `PSFMATCH = T` and `PSFMODE = 'CUBE-COMMON'`
mark a homogenized cube; native mosaics have `PSFMATCH = F` and no `PSFMODE`. Both
kinds carry `SPCVER = 'SSDC R7 202607 Table 2'`, so the current `strict_version`
check passes. Do not use `PSFMODE = 'CUBE-PLANE'` (the `*-PSFPLANE.fits` products)
for pixel SEDs: its planes have different resolutions. Reject it with a clear error.

**HDUs the importer already uses** (unchanged): `IMAGE_MJYPIX` and `ERROR_MJYPIX`
in mJy/pixel, and `SPECTRAL_CHANNELS`. Checked: `IMAGE_MJYPIX = IMAGE[MJy/sr] ×
PIXEL_AREA[sr] × 1e9` to 1.000000. `IMAGE` already has the zodiacal model removed
(`IMAGE = IMAGE_RAW − ZODI_MODEL`).

**Additional HDUs to read** (homogenized cubes only):

| HDU | Content | Use |
|---|---|---|
| `PSF_COVERAGE` (102×ny×nx) | Fraction of the PSF footprint with valid input; threshold in primary `PSFCOVTH` (0.999) | Pixels below the threshold do **not** have the target PSF. Invalidate them |
| `FLAGS` (uint32) | Bit flags; the reject mask is in the `FLAGS` HDU header as `REJMASK` (1070780101) and `REJBITS` | `FLAGS & REJMASK != 0` → invalid |
| `PLANE_QUALITY` (table, `PLQVER = '4'`) | One row per plane. `PLANE` is 0-based (= cube index). Columns `PLANEQ`, `QFLAGS`, `ERRSCALE`, `PSFQ`, `PSF_R80_EXCESS`, `SYS_FRAC`, … | Plane screening and header keys |
| `PSF_INPUTS` (table) | `TARGET_SIGMA` [pixel] (NGC 3627: 1.124 for every row; `MODE = 'CUBE-COMMON'`) | Target PSF |
| `PSF_GRID` (102×36×11×11) | Model-achieved matched PSF per plane and tile, native-sampled and pixel-integrated, sum 1, tiny negatives (~1e-7) possible. Tile boxes in `PSF_TILES` (`X0,X1,Y0,Y1`) | Spatial validation of the analytic PSF |
| `NOISE_CORR` (102×15×15) | Pixel-to-pixel noise correlation ρ(dx,dy), centre = 1 | Binned-error correction factors |
| `WAVELENGTH_EFF`, `BANDWIDTH_EFF` (102×ny×nx) [µm] | Per-pixel effective wavelength. The LVF makes it vary by about 0.5 bandwidth (P1–P99) across 5′ | Header summary only (no new image layers) |

`PLANEQ` values: `OK`, `HE1083` (He 1.083 µm airglow, **exclude**),
`HE1083_WING` and `HE1083_LEAK` (caution, keep). Empty planes exist in QR3 cubes:
all pixels NaN, listed in primary `EMPTYPL` (NGC 628 QR3: 0-based planes 8–11,
59–62). Detect them as "no finite pixel" rather than parsing `EMPTYPL`.

**The PSF of a homogenized cube.** Every plane has the same PSF, a circular
Gaussian with σ_T = `TARGET_SIGMA` native pixels, defined before pixel
integration. A native pixel value therefore sees
`PRF = Gaussian(σ_T) ⊗ box(1 native pixel)`, whose second moment is
√(σ_T² + 1/12) = 1.161 px (FWHM ≈ 16.3″ before pixel integration). Checked: this
analytic PRF matches every `PSF_GRID` tile of NGC 3627 to L1 = 0.92–1.03 %, which
is within `SafePSFConfig.spatial_l1_tolerance = 0.02`.

Known exception: planes with `PSFQ = 'MISMATCH'` (NGC 3627: D6, planes 85–101).
There the empirically measured r80 is larger than the target by
`PSF_R80_EXCESS` (≤ 0.0226).

## 3. Why the current code fails on these cubes

1. The importer sets `requires_output_effective_psf=True`. Then only
   `ExternalPSFProvider` can resolve a PSF (`psf/providers.py:291`), and the other
   providers are excluded in `PSF_safe._provider_registry`.
   * If `safe_psf_config.mosaic_synthesis` is set, `build_spherex_mosaic_psf_manifest`
     reads the `INPUTS` HDU and synthesizes the **pre-homogenization** coadd PSF.
     It never checks `PSFMATCH`. The result is a wrong (too narrow) PSF, so
     `psf_convolve` smooths an already homogenized cube again.
   * If `mosaic_synthesis` is not set and there is no manifest, the PSF lookup
     fails (`ProviderUnavailable`), and `prepare` stops the run with `PSFMatchError`.
2. Pixels with `PSF_COVERAGE < 0.999` or rejected flags are imported as valid
   data: 2.1 % of NGC 3627 and 14.0 % of NGC 628 QR3. He-line planes and
   all-NaN planes are imported as bands.
3. Per-plane quality (systematic fraction, error scale, noise correlation) is lost.

## 4. Required changes (SPHEREx-specific only)

### R1. Detection and dispatch (`utils/spherex_cube.py`)
Read the primary header:
* `PSFMATCH` true and `PSFMODE == 'CUBE-COMMON'` → homogenized path.
* `PSFMODE == 'CUBE-PLANE'` → `ValueError`.
* Otherwise → the current path, unchanged.

### R2. Plane screening (homogenized path only)
New keyword `drop_flagged_planes: bool = True` on `load()`:
* Drop planes with `PLANEQ == 'HE1083'` and planes with no finite `IMAGE_MJYPIX`
  pixel.
* If the caller passes `channels=` explicitly, respect it, but warn for every
  selected plane that would have been dropped.
* Store the dropped list and the reasons on the returned image set (for example
  `image_set._spherex_dropped_planes`) and report it through the existing
  reporting helpers.

Expected band counts: 99 for NGC 3627 (102 − 3 He); 91 for NGC 628 QR3
(102 − 3 He − 8 empty).

### R3. Pixel validity (homogenized path only)
`invalid = ~isfinite(image) | (PSF_COVERAGE < PSFCOVTH) | ((FLAGS & REJMASK) != 0)`.
Set image and error to NaN at invalid pixels. Read `PSFCOVTH` from the primary
header (default 0.999) and `REJMASK` from the `FLAGS` HDU header. If `REJMASK`
is missing, skip the flag term and warn.

### R4. Band header keys (homogenized path only; ≤ 8 characters)

| Key | Value |
|---|---|
| `SPXPSFH` | `T`: homogenized cube |
| `SPXPSFMD` | Source `PSFMODE` |
| `SPXTSIG` | `TARGET_SIGMA` [native px] |
| `SPXPLQ`, `SPXQFLG` | `PLANEQ`, `QFLAGS` of the plane |
| `SPXSYS` | `SYS_FRAC`: fractional systematic error, correlated within a detector |
| `SPXERRSC` | `ERRSCALE`: empirical error scale. **Record only; do not apply** (the producer's convention) |
| `SPXPSFQ`, `SPXR80X` | `PSFQ`, `PSF_R80_EXCESS` |
| `NCORRF2` … `NCORRF7` | Variance inflation of an n×n block sum relative to the quadrature sum, from `NOISE_CORR` (reference code in §6). NGC 3627: NCORRF3 ≈ 5.00, NCORRF5 ≈ 7.47. These are generic names, for optional change O1 |
| `NCORRSC` | Pixel scale [arcsec] at which `NCORRF*` are valid (6.15) |
| `SPXLAMEF`, `SPXLAMSP` | Median and P1–P99 spread of `WAVELENGTH_EFF` over valid pixels [µm] |

Remove the merged `PSFMATCH` and `PSFMODE` cards from the band header after
copying them to `SPXPSFH` and `SPXPSFMD`. `PSF_safe.convolution` writes its own
`PSFMATCH`/`PSFMODE`, and `utils/checkpoint.py:664` tests `PSFMODE`.

### R5. Product metadata
Keep `product_kind='coadd'` and `requires_output_effective_psf=True`. Add to
`provenance`:
* `psf_homogenized: True`
* `target_sigma_pixel`
* `psf_grid_hdu: 'PSF_GRID'`
* `psf_tiles_hdu: 'PSF_TILES'`
* `psf_quality: {psfq, r80_excess}`
* `psf_mismatch_policy`

The new importer keyword `psf_mismatch_policy: Literal['keep','broaden','drop'] = 'keep'`
controls what happens to planes with `PSFQ == 'MISMATCH'`:
* `drop`: drop those planes (R2).
* `broaden`: the provider widens σ.
* `keep`: warn only.

### R6. New PSF provider (SPHEREx handler module)
Register it with `@register_provider_factory`, following the pattern of
`_spherex_provider_factory` in `handlers/spherex_psf.py`.

* `name = "spherex-psfcommon-gaussian"`, `priority = 320`. This is higher than
  `ExternalPSFProvider` (300/310), so a stale synthesized manifest can never win.
  If a manifest entry also matches the product, emit a warning that names it.
* `match`: facility is SPHEREx, `product_kind == 'coadd'`, and
  `provenance.psf_homogenized` is true.
* `resolve`:
  1. Render the oversampled PRF with `oversample = 5` and half-size
     ≥ ceil(8 √(σ²+1/12)) native px (reference code in §6). The sampling is the
     product's native `PixelSampling` divided by 5. The stamp is odd-sized and
     centred on a native pixel centre, so Spec7DT's point-sampling resampler
     (`_resample_psf_affine`) recovers the exact native PRF on the native grid.
  2. Validate it with the injected `validator`, with source
     `spherex-psfcommon:sigma=<σ>` and provenance holding the plane, σ, the
     policy and `pixel_integrated=True`.
  3. Spatial validation: for each `PSF_GRID` tile of the plane (skip tiles that
     are all non-finite or have sum ≤ 0), compute against the native PRF the L1
     error, the maximum encircled-energy difference, and the centroid offset
     [px]. Set `ensemble.spatial_metrics` to `spatial_l1_max`, `spatial_ee_max`,
     `spatial_centroid_max` and `spatial_r80_spread`.
  4. Raise `ProviderRejected` if any metric exceeds the corresponding
     `SafePSFConfig` tolerance (`spatial_l1_tolerance`, `spatial_ee_tolerance`,
     `kernel_centroid_tolerance`, `spatial_r80_tolerance`).
  5. `broaden` policy with a finite `PSF_R80_EXCESS = e`: solve for σ′ with
     r80(PRF σ′) = (1+e)·r80(PRF σ_T) by bisection, and use σ′. Record both σ
     values.
  6. Return `ResolvedPSF(model, ensemble, self.name)`. Use `singleton_ensemble`
     plus `spatial_metrics`, or an equivalent that the rest of `PSF_safe` already
     accepts.

### R7. Never synthesize for homogenized cubes
In `handlers/spherex_mosaic_psf._group_products`, skip products whose
`provenance.psf_homogenized` is true. Then `spherex_mosaic_synthesis_required`
ignores them, and no IRSA query or download happens for them.

## 5. Generic changes (small, header-driven; all approved by the user, see §8)

* **O1 — correlated noise in binning** (`division/binning.py`): multiply
  `binned_err` by `sqrt(header[f"NCORRF{bin_size}"])` when the key exists and
  `NCORRSC` equals the current pixel scale. Otherwise leave the value unchanged
  and warn once. Without O1, SPHEREx bin errors are low by ×2.24 (3×3) and
  ×2.73 (5×5).
* **O2 — invalid pixels after unit conversion** (`utils/unit.py`):
  `unitConvertor` turns NaN data into 0 but leaves the error NaN, so a bin that
  touches an invalid pixel gets a low flux. Follow the existing Gaia-mask
  convention (`error_fill = 999`): where data was NaN, set the error to the fill
  value.
* **O3 — systematic error in the catalog** (`utils/file_generator.py`): if a
  band header has `SPXSYS`, add a `{flux}_sys = SPXSYS·|flux|` column, or document
  that the user combines it in quadrature after the catalog is made.
* **O4 — background mode per observatory** (`reduction/background.py`): for
  example `BackgroundConfig(mode_by_observatory={"SPHEREx": "constant"})`. These
  cubes are zodi-subtracted; only a constant residual offset should be removed.

Not code changes, but documentation for users: SWarp registration
(`default.swarp`: `NEAREST`, `CENTER_TYPE ALL`, `PIXELSCALE_TYPE MAX`) leaves
SPHEREx unresampled only when the SPHEREx field contains every other image and
has the largest pixel. Otherwise SPHEREx can shift by up to 0.5 output pixel.
Add a check that compares the registered SPHEREx arrays with the imported ones.

## 6. Reference code (tested in the producer repository)

```python
import numpy as np
from scipy.special import erf

def spherex_prf(sigma_px: float, oversample: int = 5) -> np.ndarray:
    """Gaussian(sigma_px) convolved with a 1-native-pixel box, sampled at
    sub-pixel centres; odd size, centred on a native pixel centre, sum 1."""
    half = int(np.ceil(8.0 * np.hypot(sigma_px, np.sqrt(1.0 / 12.0))))
    n = (2 * half + 1) * oversample
    x = (np.arange(n) - (n - 1) / 2.0) / oversample          # native-pixel units
    s = np.sqrt(2.0) * sigma_px
    e = 0.5 * (erf((x + 0.5) / s) - erf((x - 0.5) / s))
    prf = np.outer(e, e)
    return prf / prf.sum()

def block_correlation_factor(noise_corr: np.ndarray, size: int) -> float:
    """Var(sum of a size x size block) / (size^2 sigma^2); 1 = uncorrelated."""
    if not np.isfinite(noise_corr).any():
        return 1.0
    c = noise_corr.shape[0] // 2
    total = 0.0
    for dy in range(-(size - 1), size):
        for dx in range(-(size - 1), size):
            if abs(dy) <= c and abs(dx) <= c and np.isfinite(noise_corr[c + dy, c + dx]):
                total += (size - abs(dx)) * (size - abs(dy)) * noise_corr[c + dy, c + dx]
    return max(total / size ** 2, 1.0)
```

The native PRF is `spherex_prf(σ, 1)`. On the native grid the oversampled stamp
taken every 5th sample from the centre equals it up to normalization.

## 7. Acceptance tests (add to `test/`, e.g. `test_spherex_cube_psfcommon.py`)

* **T1 — native regression:** importing the native mosaic gives arrays, headers
  and product metadata identical to v0.15.0 before the change.
* **T2 — NGC 3627 PSFCOMMON:**
  * 99 bands; band of plane 17 has `SPXPLQ = 'HE1083_WING'`.
  * NaN fraction equals the R3 mask.
  * On valid pixels, data equals `IMAGE × PIXEL_AREA × 1e9`.
  * `NCORRF3` within 4.7–5.2.
  * `PSFMATCH`/`PSFMODE` are absent from band headers.
* **T3 — NGC 628 QR3 PSFCOMMON:** 91 bands; planes 8–11, 59–62 and 14–16 reported
  as dropped, with reasons.
* **T4 — provider:**
  * The PSF resolves for every band of T2 through `spherex-psfcommon-gaussian`.
  * The native-sampled second moment is 1.161 ± 0.002 px.
  * The `PSF_GRID` L1 maximum is ≤ 0.0105.
  * Validator edge mass < `psf_edge_mass_tolerance`.
* **T5 — safe matching:** for an image set made of T2 bands only, `prepare` gives
  `safety_sigma_arcsec = 0`, and every SPHEREx kernel is a delta (L1 < 1e-6).
  Add a synthetic band with a 6″ Gaussian PSF: the SPHEREx kernels stay delta
  whenever the chosen safety σ is 0.
* **T6 — no synthesis:** with `mosaic_synthesis` configured and network functions
  monkeypatched to raise, a run up to `psf_prepare` on T2 succeeds.
* **T7 — PSFPLANE rejected:** a cube with `PSFMODE = 'CUBE-PLANE'` raises `ValueError`.

## 8. Decisions made by the user (2026-11-13)

1. `psf_mismatch_policy` default is **`keep`**. `PSF_MISMATCH` planes stay in the
   image set with the target PSF, a warning, and the `SPXPSFQ`/`SPXR80X` keys.
   `broaden` and `drop` remain available as options.
2. `HE1083_WING` and `HE1083_LEAK` planes are **not** dropped. Only `HE1083`
   and empty planes are dropped by default.
3. **O1–O4 are all approved** and are part of this change. Implement them after
   R1–R7, each with its own test:
   * O1: a 3×3 bin of a T2 band has error = quadrature error × √NCORRF3.
   * O2: a bin touching an invalid pixel never has a finite error below the fill
     value.
   * O3: a `{flux}_sys` column exists for SPHEREx bands and is absent for others.
   * O4: `mode_by_observatory={"SPHEREx": "constant"}` leaves other observatories
     on `auto`.

   The SWarp note at the end of §5 stays documentation only, plus the
   registered-equals-imported check.

Still present the plan and diff to the user before editing (AGENT.md).
