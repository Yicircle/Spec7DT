# PSF-homogenized SPHEREx mosaic cubes

Spec7DT imports PSF-homogenized SPHEREx mosaic cubes (`*-PSFCOMMON.fits`, primary
header `PSFMATCH = T`, `PSFMODE = 'CUBE-COMMON'`) through the same entry points as
native mosaics:

```python
from Spec7DT.utils.file_handler import ImageQuery

image_set = ImageQuery.querySpherexCube(
    "NGC3627-SPHEREx-MOSAIC-PSFCOMMON.fits",
    drop_flagged_planes=True,      # default
    psf_mismatch_policy="keep",    # default; or "broaden" / "drop"
)
```

Native mosaics (`PSFMATCH = F`) are imported exactly as before. Cubes with
`PSFMODE = 'CUBE-PLANE'` (`*-PSFPLANE.fits`) are rejected, because their planes
have different resolutions and cannot form pixel SEDs.

## What the importer does

* **Plane screening.** Planes with `PLANEQ = 'HE1083'` (He 1.083 µm airglow) and
  planes without any finite pixel are dropped. `HE1083_WING` and `HE1083_LEAK`
  planes are kept. If you pass `channels=` explicitly, your selection is kept and
  a warning names every selected plane that would otherwise have been dropped.
  Dropped planes and reasons are stored in `image_set._spherex_dropped_planes`.
* **Pixel validity.** A pixel is set to NaN in both data and error when it is
  non-finite, when `PSF_COVERAGE < PSFCOVTH` (it does not have the target PSF),
  or when `FLAGS & REJMASK != 0`.
* **PSF.** Every plane has the same PSF: a circular Gaussian of
  `TARGET_SIGMA` native pixels convolved with the 6.15″ pixel. The
  `spherex-psfcommon-gaussian` provider renders it and validates it against the
  cube's `PSF_GRID` tiles. No PSF is synthesized from the Level-2 exposures, so
  `SafePSFConfig.mosaic_synthesis` never triggers an IRSA query for these cubes.
  When SPHEREx has the broadest PSF of the image set, its matching kernels are
  exact deltas.
* **`PSFQ = 'MISMATCH'` planes** (measured r80 larger than the target by
  `PSF_R80_EXCESS`): `keep` uses the target PSF and warns; `broaden` widens the
  Gaussian so that r80 grows by `PSF_R80_EXCESS`; `drop` removes the planes.

## Band header keys

| Key | Meaning |
|---|---|
| `SPXPSFH`, `SPXPSFMD` | Homogenized cube flag and source `PSFMODE` (the cube's `PSFMATCH`/`PSFMODE` cards are removed) |
| `SPXTSIG` | Target Gaussian sigma [native px] |
| `SPXPLQ`, `SPXQFLG` | Plane quality `PLANEQ` and `QFLAGS` |
| `SPXSYS` | Fractional systematic error `SYS_FRAC` (see below) |
| `SPXERRSC` | Empirical error scale `ERRSCALE`; recorded only, **not applied** |
| `SPXPSFQ`, `SPXR80X` | `PSFQ` and `PSF_R80_EXCESS` |
| `NCORRF2` … `NCORRF7` | Variance factor of an n×n block sum relative to the quadrature sum, from `NOISE_CORR` |
| `NCORRSC` | Pixel scale [arcsec] at which `NCORRF*` are valid |
| `SPXLAMEF`, `SPXLAMSP` | Median and P1–P99 spread of the per-pixel effective wavelength [µm] |

## Errors in the SED catalog

* **Correlated noise.** Pixel binning multiplies the binned error by
  `sqrt(NCORRF<n>)` when the band header carries the factor and the pixel scale
  still equals `NCORRSC`. Otherwise the error is left unchanged and a warning is
  emitted. The factors are removed when a non-delta PSF-matching kernel
  convolves the band, because convolution changes the noise correlation.
* **Invalid pixels.** After unit conversion, pixels without a valid measurement
  have flux 0 and error 999 (the same fill value as the Gaia mask), so a bin
  that touches one never reports a small error.
* **Systematic error (`SPXSYS`).** The catalog columns contain statistical
  errors only. `SYS_FRAC` is a fractional systematic error that is correlated
  within a detector, so it must not be averaged down by binning or summing.
  Combine it yourself after the catalog is written:

  ```python
  sigma_total = np.sqrt(sigma_stat**2 + (SPXSYS * np.abs(flux))**2)
  ```

  Use the `SPXSYS` value of the corresponding band header (one value per
  plane). For integrated (total) fluxes, add the systematic term linearly
  across pixels of the same plane before combining it in quadrature.
* **Error scale (`SPXERRSC`).** Following the producer's convention, the
  empirical error scale is recorded but not applied.

## Background subtraction

These cubes are zodi-subtracted (`IMAGE = IMAGE_RAW − ZODI_MODEL`). Remove only a
constant residual offset for SPHEREx while other observatories keep automatic
model selection:

```python
background_config = {"mode_by_observatory": {"SPHEREx": "constant"}}
```

## Registration (SWarp)

The default SWarp configuration (`RESAMPLING_TYPE NEAREST`, `CENTER_TYPE ALL`,
`PIXELSCALE_TYPE MAX`) leaves SPHEREx unresampled only when the SPHEREx field
contains every other image and has the largest pixel. Otherwise SPHEREx can be
shifted by up to 0.5 output pixel or resampled to another pixel scale, and its
PSF and noise-correlation metadata no longer describe the registered image.
Spec7DT checks every homogenized SPHEREx band after registration and warns if
the registered grid or pixel values differ from the imported ones. Trim the
other images to the SPHEREx footprint, or make SPHEREx the coarsest image, to
avoid this.
