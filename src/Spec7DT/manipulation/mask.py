import math
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.time import Time
from astropy.wcs import WCS
from photutils.segmentation import detect_sources
from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from photutils.psf import PSFPhotometry, MoffatPSF
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.ndimage import find_objects

from ..utils.utility import useful_functions
from ..utils.reporting import emit_alert, emit_detail

class Masking:
    _MANUAL_MASK_COLUMNS = ("_RAJ2000", "_DEJ2000", "radius", "matched_galaxy")
    _RA_COLUMNS = ("_RAJ2000", "RA_ICRS", "RAJ2000", "RA", "ra", "ra_deg")
    _DEC_COLUMNS = ("_DEJ2000", "DE_ICRS", "DEJ2000", "DEC", "dec", "dec_deg")
    _GMAG_COLUMNS = ("Gmag", "phot_g_mean_mag")
    _PLX_COLUMNS = ("Plx", "plx", "parallax")
    _E_PLX_COLUMNS = ("e_Plx", "e_plx", "parallax_error")
    _PM_COLUMNS = ("PM", "pm")
    _E_PM_COLUMNS = ("e_PM", "pm_error")
    _PMRA_COLUMNS = ("pmRA", "pmra")
    _E_PMRA_COLUMNS = ("e_pmRA", "pmra_error")
    _PMDEC_COLUMNS = ("pmDE", "pmdec")
    _E_PMDEC_COLUMNS = ("e_pmDE", "pmdec_error")
    _QSO_COLUMNS = ("QSO", "qso", "in_qso_candidates")
    _GAL_COLUMNS = ("Gal", "gal", "in_galaxy_candidates")
    _RUWE_COLUMNS = ("RUWE", "ruwe")
    _SOURCE_ID_COLUMNS = ("source_id", "SOURCE_ID", "Source")
    _REF_EPOCH_COLUMNS = ("ref_epoch", "REF_EPOCH", "Epoch")

    def __init__(self):
        pass
    
    @classmethod
    def adapt_mask(cls, image_data, header, error_data, galaxy_name, observatory, band, image_set,
                   manual, gaia_mask=None, metadata_resolver=None):
        gaia_config = cls._gaia_config(gaia_mask)
        debug_plot = gaia_config is not None and gaia_config.get("debug_plot", False)
        debug_sources = {"masked": [], "rejected": []} if debug_plot else None
        reference = None
        if gaia_config is not None:
            reference = cls.prepare_gaia_reference(
                image_data=image_data,
                header=header,
                error_data=error_data,
                galaxy_name=galaxy_name,
                observatory=observatory,
                band=band,
                image_set=image_set,
                gaia_mask=gaia_config,
                metadata_resolver=metadata_resolver,
            )
        reference_sources = cls.reference_sources_for_target(
            image_set,
            galaxy_name,
            observatory,
            band,
            gaia_config,
            reference,
        )
        try:
            fwhm_arcsec = cls._target_psf_fwhm_arcsec(image_set, galaxy_name, gaia_config)
        except ValueError as exc:
            warnings.warn(
                f"Skipping masking for {galaxy_name}/{observatory}/{band}: {exc}",
                RuntimeWarning,
            )
            return
        pixel_scale = useful_functions.get_pixel_scale(header)
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError(
                f"Invalid pixel scale for {galaxy_name}/{observatory}/{band}: {pixel_scale}"
            )
        if manual is not None:
            try:
                WCS(header)
            except Exception as exc:
                target = f"{galaxy_name}/{observatory}/{band}"
                raise ValueError(
                    f"Could not construct WCS for manual mask target {target}: {exc}"
                ) from exc
        fwhm = fwhm_arcsec / pixel_scale  # in pixel
        
        mask_image, masked_image, _ = cls.make_mask(
            cls,
            image_data,
            header,
            galaxy_name,
            fwhm,
            manual,
            gaia_mask=gaia_config,
            metadata_resolver=metadata_resolver,
            debug_sources=debug_sources,
            reference_sources=reference_sources,
            observatory=observatory,
            band=band,
        )
        if debug_plot:
            cls.plot_mask_debug(mask_image, masked_image, debug_sources)

        error_fill = gaia_config.get("error_fill", 999.0) if gaia_config is not None else 999.0
        masked_err = np.where(mask_image, error_fill, error_data)
        
        image_set.update_data(masked_image, galaxy_name, observatory, band)
        image_set.update_error(masked_err, galaxy_name, observatory, band)

    @classmethod
    def _target_psf_fwhm_arcsec(cls, image_set, galaxy_name, gaia_config=None):
        from ..reduction.PSF import PointSpreadFunction

        fallback = None if gaia_config is None else gaia_config.get("fallback_fwhm_arcsec")
        try:
            return PointSpreadFunction.target_fwhm(image_set, galaxy_name)
        except ValueError as exc:
            if fallback is not None:
                return float(fallback)
            raise ValueError("No valid PSF FWHM values are available for masking.") from exc

    @classmethod
    def prepare_gaia_reference(cls, image_data, header, error_data, galaxy_name, observatory,
                               band, image_set, gaia_mask=None, metadata_resolver=None):
        gaia_config = cls._gaia_config(gaia_mask)
        if gaia_config is None:
            return None

        mode = gaia_config.get("reference_detection_mode", "auto")
        if mode not in {"auto", "off", "required"}:
            raise ValueError("reference_detection_mode must be 'auto', 'off', or 'required'.")

        cache = getattr(image_set, "_gaia_mask_reference_cache", None)
        if cache is None:
            cache = {}
            setattr(image_set, "_gaia_mask_reference_cache", cache)
        if galaxy_name in cache:
            return cache[galaxy_name]

        if mode == "off":
            result = {"status": "legacy", "reason": "reference detection disabled"}
            cache[galaxy_name] = result
            return result

        try:
            result = cls.build_gaia_reference(
                image_set,
                galaxy_name,
                gaia_config,
                metadata_resolver=metadata_resolver,
            )
        except Exception as exc:
            if mode == "required":
                raise
            result = {"status": "legacy", "reason": str(exc)}

        if result.get("status") != "ready" and mode == "required":
            raise ValueError(f"Could not prepare required GAIA mask reference: {result.get('reason')}")

        cache[galaxy_name] = result
        if result.get("status") != "ready":
            if gaia_config.get("warn", True):
                warnings.warn(
                    "Could not prepare optical GAIA mask reference; using legacy per-band "
                    f"masking: {result.get('reason', 'unknown reason')}"
                )
            return result

        cls.report_reference_debug(result, gaia_config)
        if gaia_config.get("debug_plot", False):
            cls.plot_reference_debug(result)
        return result

    @classmethod
    def build_gaia_reference(cls, image_set, galaxy_name, gaia_config, metadata_resolver=None):
        candidates = cls.reference_candidates(image_set, galaxy_name, gaia_config)
        if not candidates:
            return {"status": "legacy", "reason": "no optical image with a known wavelength"}

        sky_region = cls.reference_sky_region(candidates, gaia_config)
        first = candidates[0]
        catalog = cls.load_gaia_catalog(
            cls,
            first["image"],
            first["header"],
            gaia_config,
            sky_region=sky_region,
        )
        sources = cls.normalize_gaia_catalog(cls, catalog)
        if not sources:
            return {"status": "legacy", "reason": "no usable GAIA catalog sources"}

        requested_target = gaia_config.get("reference_target")
        evaluated = {}
        if requested_target is not None:
            selected_candidate = cls.find_requested_reference(candidates, requested_target)
            selected = cls.evaluate_reference_candidate(selected_candidate, sources, gaia_config)
            evaluated[selected_candidate["key"]] = selected
            selection_reason = "explicit reference target"
        else:
            minimum_matches = max(1, int(gaia_config.get("reference_min_matches", 3)))
            eligible = []
            for candidate in candidates:
                if not candidate["error_valid"]:
                    continue
                try:
                    evaluation = cls.evaluate_reference_candidate(candidate, sources, gaia_config)
                except Exception as exc:
                    if gaia_config.get("debug", False):
                        warnings.warn(
                            f"Could not evaluate GAIA reference candidate {candidate['key']}: {exc}",
                            UserWarning,
                        )
                    continue
                evaluated[candidate["key"]] = evaluation
                if evaluation["match_count"] >= minimum_matches and np.isfinite(evaluation["snr_score"]):
                    eligible.append(evaluation)

            if eligible:
                fallback_wave = float(gaia_config.get("reference_fallback_angstrom", 6000.0))
                selected = max(
                    eligible,
                    key=lambda item: (
                        item["snr_score"],
                        item["match_count"],
                        -abs(item["candidate"]["wavelength"] - fallback_wave),
                    ),
                )
                selection_reason = "highest matched-star S/N"
            else:
                fallback_wave = float(gaia_config.get("reference_fallback_angstrom", 6000.0))
                selected_candidate = min(
                    candidates,
                    key=lambda item: abs(item["wavelength"] - fallback_wave),
                )
                selected = evaluated.get(selected_candidate["key"])
                if selected is None:
                    selected = cls.evaluate_reference_candidate(selected_candidate, sources, gaia_config)
                    evaluated[selected_candidate["key"]] = selected
                selection_reason = "closest wavelength fallback"

        matched_records = cls.add_reference_radii(selected, gaia_config)
        matched_indices = {record["gaia_index"] for record in matched_records}
        fallback_records = cls.unmatched_gaia_records(
            sources,
            matched_indices,
            selected["candidate"]["header"],
            gaia_config,
        )
        summaries = []
        for evaluation in evaluated.values():
            summaries.append({
                "observatory": evaluation["candidate"]["observatory"],
                "band": evaluation["candidate"]["band"],
                "wavelength": evaluation["candidate"]["wavelength"],
                "error_valid": evaluation["candidate"]["error_valid"],
                "detections": len(evaluation["detections"]),
                "matches": evaluation["match_count"],
                "snr_score": evaluation["snr_score"],
            })

        candidate = selected["candidate"]
        return {
            "status": "ready",
            "reason": selection_reason,
            "galaxy": galaxy_name,
            "observatory": candidate["observatory"],
            "band": candidate["band"],
            "wavelength": candidate["wavelength"],
            "snr_score": selected["snr_score"],
            "match_radius_arcsec": selected["match_radius_arcsec"],
            "catalog_rows": cls.catalog_length(cls, catalog),
            "normalized_sources": len(sources),
            "candidate_summaries": summaries,
            "detections": selected["detections"],
            "matched_sources": matched_records,
            "fallback_sources": fallback_records,
            "mask_sources": matched_records + fallback_records,
            "reference_image": candidate["image"],
            "reference_header": candidate["header"],
        }

    @classmethod
    def reference_candidates(cls, image_set, galaxy_name, gaia_config):
        galaxy_data = getattr(image_set, "data", {}).get(galaxy_name, {})
        optical_range = gaia_config.get("reference_optical_range_angstrom", (3500.0, 10000.0))
        try:
            wave_min, wave_max = map(float, optical_range)
        except Exception as exc:
            raise ValueError("reference_optical_range_angstrom must contain two values") from exc
        requested_target = gaia_config.get("reference_target")
        requested_key = tuple(requested_target) if requested_target is not None else None

        candidates = []
        for obs, bands in galaxy_data.items():
            for candidate_band, image in bands.items():
                key = (obs, candidate_band)
                try:
                    curve = image_set.filter_inst.get_filter(name=candidate_band, facility=obs)
                    wavelength = float(curve.pivot_wavelength)
                except Exception:
                    if requested_key != key:
                        continue
                    wavelength = float(gaia_config.get("reference_fallback_angstrom", 6000.0))
                if requested_key != key and not (wave_min <= wavelength <= wave_max):
                    continue

                header = image_set.header[galaxy_name][obs][candidate_band]
                error = getattr(image_set, "error", {}).get(galaxy_name, {}).get(obs, {}).get(candidate_band)
                error_valid, error_fraction = cls.valid_reference_error(image, error, gaia_config)
                psf_arcsec = cls.reference_psf_fwhm(image_set, galaxy_name, obs, candidate_band, gaia_config)
                candidates.append({
                    "key": key,
                    "observatory": obs,
                    "band": candidate_band,
                    "wavelength": wavelength,
                    "image": np.asarray(image),
                    "header": header,
                    "error": None if error is None else np.asarray(error),
                    "error_valid": error_valid,
                    "error_fraction": error_fraction,
                    "psf_arcsec": psf_arcsec,
                })
        return candidates

    @staticmethod
    def find_requested_reference(candidates, requested_target):
        try:
            target = tuple(requested_target)
        except Exception as exc:
            raise ValueError("reference_target must be (observatory, band)") from exc
        if len(target) != 2:
            raise ValueError("reference_target must be (observatory, band)")
        for candidate in candidates:
            if candidate["key"] == target:
                return candidate
        raise ValueError(f"Requested reference image is unavailable: {target}")

    @staticmethod
    def valid_reference_error(image, error, gaia_config):
        if error is None or np.shape(error) != np.shape(image):
            return False, 0.0
        image = np.asarray(image)
        error = np.asarray(error)
        science_valid = np.isfinite(image) & (image != 0)
        science_count = int(np.sum(science_valid))
        if science_count == 0:
            return False, 0.0
        error_valid = science_valid & np.isfinite(error) & (error > 0)
        fraction = float(np.sum(error_valid) / science_count)
        minimum = float(gaia_config.get("reference_min_error_fraction", 0.5))
        return fraction >= minimum, fraction

    @classmethod
    def reference_psf_fwhm(cls, image_set, galaxy, observatory, band, gaia_config):
        try:
            value = float(image_set.psf[galaxy][observatory][band])
            if np.isfinite(value) and value > 0:
                return value
        except Exception:
            pass
        fallback = gaia_config.get("fallback_fwhm_arcsec")
        if fallback is not None and np.isfinite(float(fallback)) and float(fallback) > 0:
            return float(fallback)
        values = useful_functions.extract_values_recursive(getattr(image_set, "psf", {}), galaxy)
        numeric = []
        for value in values:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                numeric.append(value)
        if numeric:
            return float(np.nanmedian(numeric))
        raise ValueError(f"No valid PSF is available for {galaxy}/{observatory}/{band}")

    @classmethod
    def reference_sky_region(cls, candidates, gaia_config):
        corners = []
        for candidate in candidates:
            image = candidate["image"]
            ny, nx = image.shape
            pixels = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]], dtype=float)
            sky = WCS(candidate["header"]).pixel_to_world(pixels[:, 0], pixels[:, 1])
            sky = sky if isinstance(sky, SkyCoord) else SkyCoord(sky)
            corners.append(sky.icrs)
        all_sky = SkyCoord(
            ra=np.concatenate([coord.ra.deg for coord in corners]) * u.deg,
            dec=np.concatenate([coord.dec.deg for coord in corners]) * u.deg,
            frame="icrs",
        )
        cartesian = all_sky.cartesian
        center = SkyCoord(
            x=np.mean(cartesian.x),
            y=np.mean(cartesian.y),
            z=np.mean(cartesian.z),
            representation_type="cartesian",
            frame="icrs",
        )
        margin = float(gaia_config.get("query_radius_margin", 1.05))
        radius = np.nanmax(center.separation(all_sky)) * margin
        return center, radius

    @classmethod
    def evaluate_reference_candidate(cls, candidate, sources, gaia_config):
        image = np.asarray(candidate["image"], dtype=float)
        coverage_mask = ~np.isfinite(image) | (image == 0)
        finite = image[~coverage_mask]
        if finite.size == 0:
            raise ValueError(f"Reference image has no finite coverage: {candidate['key']}")

        try:
            ny, nx = image.shape
            box_size = (max(3, min(ny, ny // 4)), max(3, min(nx, nx // 4)))
            background = Background2D(
                image,
                box_size=box_size,
                filter_size=(3, 3),
                coverage_mask=coverage_mask,
                fill_value=0.0,
                bkg_estimator=MedianBackground(),
                exclude_percentile=50.0,
            )
            background_map = np.asarray(background.background, dtype=float)
            background_rms = np.asarray(background.background_rms, dtype=float)
        except Exception:
            _, median, std = sigma_clipped_stats(finite, sigma=3.0)
            if not np.isfinite(std) or std <= 0:
                std = np.nanstd(finite)
            std = std if np.isfinite(std) and std > 0 else np.finfo(float).eps
            background_map = np.full_like(image, median, dtype=float)
            background_rms = np.full_like(image, std, dtype=float)

        positive_rms = background_rms[np.isfinite(background_rms) & (background_rms > 0)]
        rms_fallback = np.nanmedian(positive_rms) if positive_rms.size else np.nanstd(finite)
        if not np.isfinite(rms_fallback) or rms_fallback <= 0:
            rms_fallback = np.finfo(float).eps
        background_rms = np.where(
            np.isfinite(background_rms) & (background_rms > 0),
            background_rms,
            rms_fallback,
        )
        if candidate["error_valid"]:
            error = np.asarray(candidate["error"], dtype=float)
            error = np.where(np.isfinite(error) & (error > 0), error, background_rms)
            noise = np.maximum(error, background_rms)
        else:
            noise = background_rms

        signal = image - background_map
        snr_image = np.divide(signal, noise, out=np.zeros_like(signal), where=noise > 0)
        snr_image[coverage_mask | ~np.isfinite(snr_image)] = 0.0
        pixel_scale = useful_functions.get_pixel_scale(candidate["header"])
        fwhm_pix = max(1.0, float(candidate["psf_arcsec"]) / pixel_scale)
        finder = DAOStarFinder(
            fwhm=fwhm_pix,
            threshold=float(gaia_config.get("detection_sigma", 5.0)),
        )
        try:
            table = finder(snr_image, mask=coverage_mask)
        except Exception as exc:
            if gaia_config.get("debug", False):
                warnings.warn(
                    f"Reference source detection failed for {candidate['key']}: {exc}",
                    UserWarning,
                )
            table = None

        detections = []
        if table is not None:
            positions = np.array(
                [(float(row["xcentroid"]), float(row["ycentroid"])) for row in table],
                dtype=float,
            )
            if positions.size:
                aperture = CircularAperture(positions, r=max(1.0, 1.5 * fwhm_pix))
                photometry = aperture_photometry(
                    signal,
                    aperture,
                    error=noise,
                    mask=coverage_mask,
                )
                wcs = WCS(candidate["header"])
                sky = wcs.pixel_to_world(positions[:, 0], positions[:, 1])
                sky = sky if isinstance(sky, SkyCoord) else SkyCoord(sky)
                for index, (x, y) in enumerate(positions):
                    flux = float(photometry["aperture_sum"][index])
                    flux_error = float(photometry["aperture_sum_err"][index])
                    snr = flux / flux_error if np.isfinite(flux_error) and flux_error > 0 else np.nan
                    detections.append({
                        "x": x,
                        "y": y,
                        "ra": float(sky[index].icrs.ra.deg),
                        "dec": float(sky[index].icrs.dec.deg),
                        "snr": float(snr),
                    })

        matches, match_radius = cls.match_reference_sources(
            detections,
            sources,
            candidate["header"],
            candidate["psf_arcsec"],
            gaia_config,
        )
        matched_snr = np.array([record["snr"] for record in matches], dtype=float)
        matched_snr = matched_snr[np.isfinite(matched_snr) & (matched_snr > 0)]
        score = float(np.nanmedian(matched_snr)) if matched_snr.size else np.nan
        return {
            "candidate": candidate,
            "detections": detections,
            "matches": matches,
            "match_count": len(matches),
            "snr_score": score,
            "match_radius_arcsec": match_radius,
        }

    @classmethod
    def match_reference_sources(cls, detections, sources, header, psf_arcsec, gaia_config):
        configured_radius = gaia_config.get("crossmatch_radius_arcsec")
        if configured_radius is None:
            match_radius = float(np.clip(max(1.0, 0.5 * float(psf_arcsec)), 1.0, 3.0))
        else:
            match_radius = float(configured_radius)
        if match_radius <= 0:
            raise ValueError("crossmatch_radius_arcsec must be positive")
        if not detections or not sources:
            return [], match_radius

        usable_indices = [
            index for index, source in enumerate(sources)
            if not cls.flag_true(cls, source.get("qso")) and not cls.flag_true(cls, source.get("gal"))
        ]
        if not usable_indices:
            return [], match_radius
        detection_sky = SkyCoord(
            ra=[item["ra"] for item in detections] * u.deg,
            dec=[item["dec"] for item in detections] * u.deg,
            frame="icrs",
        )
        propagated = [cls.source_skycoord(source=sources[index], header=header) for index in usable_indices]
        gaia_sky = SkyCoord(
            ra=[coord.icrs.ra.deg for coord in propagated] * u.deg,
            dec=[coord.icrs.dec.deg for coord in propagated] * u.deg,
            frame="icrs",
        )
        det_indices, local_gaia_indices, separations, _ = search_around_sky(
            detection_sky,
            gaia_sky,
            match_radius * u.arcsec,
        )
        order = np.argsort(separations.to_value(u.arcsec))
        used_detections = set()
        used_gaia = set()
        matches = []
        for pair_index in order:
            detection_index = int(det_indices[pair_index])
            gaia_index = usable_indices[int(local_gaia_indices[pair_index])]
            if detection_index in used_detections or gaia_index in used_gaia:
                continue
            used_detections.add(detection_index)
            used_gaia.add(gaia_index)
            detection = detections[detection_index]
            matches.append({
                "kind": "matched",
                "gaia_index": gaia_index,
                "source": sources[gaia_index],
                "x": detection["x"],
                "y": detection["y"],
                "ra": detection["ra"],
                "dec": detection["dec"],
                "snr": detection["snr"],
                "separation_arcsec": float(separations[pair_index].to_value(u.arcsec)),
            })
        return matches, match_radius

    @classmethod
    def source_skycoord(cls, source, header=None):
        coord = SkyCoord(ra=float(source["ra"]) * u.deg, dec=float(source["dec"]) * u.deg, frame="icrs")
        ref_epoch = source.get("ref_epoch", np.nan)
        pmra = source.get("pmra", np.nan)
        pmdec = source.get("pmdec", np.nan)
        target_time = cls.header_observation_time(header)
        if target_time is None or not all(np.isfinite(value) for value in (ref_epoch, pmra, pmdec)):
            return coord
        try:
            moving = SkyCoord(
                ra=float(source["ra"]) * u.deg,
                dec=float(source["dec"]) * u.deg,
                pm_ra_cosdec=float(pmra) * u.mas / u.yr,
                pm_dec=float(pmdec) * u.mas / u.yr,
                obstime=Time(float(ref_epoch), format="jyear"),
                frame="icrs",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return moving.apply_space_motion(new_obstime=target_time).icrs
        except Exception:
            return coord

    @staticmethod
    def header_observation_time(header):
        if header is None:
            return None
        for key, kwargs in (("DATE-OBS", {}), ("MJD-OBS", {"format": "mjd"})):
            value = header.get(key)
            if value is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return Time(value, **kwargs)
            except Exception:
                continue
        return None

    @classmethod
    def add_reference_radii(cls, evaluation, gaia_config):
        records = [record.copy() for record in evaluation["matches"]]
        if not records:
            return records
        candidate = evaluation["candidate"]
        positions = np.array([(record["x"], record["y"]) for record in records], dtype=float)
        pixel_scale = useful_functions.get_pixel_scale(candidate["header"])
        fwhm_pix = max(1.0, float(candidate["psf_arcsec"]) / pixel_scale)
        for index, record in enumerate(records):
            nearest = np.inf
            if len(positions) > 1:
                distances = np.hypot(positions[:, 0] - record["x"], positions[:, 1] - record["y"])
                distances[index] = np.inf
                nearest = np.nanmin(distances)
            radius = cls.estimate_gaia_mask_radius(
                cls,
                candidate["image"],
                record["x"],
                record["y"],
                fwhm_pix,
                nearest,
                pixel_scale,
                gaia_config,
                radius_mode="standard",
            )
            if radius is None or radius <= 0:
                radius = max(1.0, fwhm_pix * gaia_config.get("min_radius_fwhm", 1.0))
            record["reference_radius_arcsec"] = float(radius * pixel_scale)
        return records

    @classmethod
    def unmatched_gaia_records(cls, sources, matched_indices, header, gaia_config):
        policy = gaia_config.get("unmatched_gaia_policy", "significant_or_bright")
        if policy not in {"significant_or_bright", "matched_only", "all"}:
            raise ValueError(
                "unmatched_gaia_policy must be 'significant_or_bright', 'matched_only', or 'all'."
            )
        if policy == "matched_only":
            return []

        records = []
        for index, source in enumerate(sources):
            if index in matched_indices:
                continue
            if cls.flag_true(cls, source.get("qso")) or cls.flag_true(cls, source.get("gal")):
                continue
            if policy != "all" and not cls.is_strong_gaia_fallback(source, gaia_config):
                continue
            coord = cls.source_skycoord(source, header=header)
            records.append({
                "kind": "gaia_fallback",
                "gaia_index": index,
                "source": source,
                "ra": float(coord.ra.deg),
                "dec": float(coord.dec.deg),
                "reference_radius_arcsec": None,
            })
        return records

    @classmethod
    def is_strong_gaia_fallback(cls, source, gaia_config):
        plx = source.get("plx", np.nan)
        e_plx = source.get("e_plx", np.nan)
        if np.isfinite(plx) and np.isfinite(e_plx) and e_plx > 0:
            if abs(plx / e_plx) >= gaia_config.get("parallax_sigma", 3.0):
                return True
        pm_significance = cls.proper_motion_significance(cls, source)
        if np.isfinite(pm_significance) and pm_significance >= gaia_config.get("pm_sigma", 3.0):
            return True
        gmag = source.get("gmag", np.nan)
        gmag_limit = gaia_config.get("unmatched_gaia_gmag_max", 15.0)
        return gmag_limit is not None and np.isfinite(gmag) and gmag <= float(gmag_limit)

    @classmethod
    def reference_sources_for_target(cls, image_set, galaxy, observatory, band, gaia_config, reference):
        if gaia_config is None:
            return None
        maximum_wavelength = gaia_config.get("mask_max_wavelength_angstrom")
        if maximum_wavelength is not None:
            try:
                curve = image_set.filter_inst.get_filter(name=band, facility=observatory)
                if float(curve.pivot_wavelength) > float(maximum_wavelength):
                    return []
            except Exception:
                if gaia_config.get("warn", True):
                    warnings.warn(
                        f"Could not resolve wavelength for {observatory}.{band}; applying GAIA mask."
                    )
        if reference is None or reference.get("status") != "ready":
            return None
        return reference.get("mask_sources", [])

    @staticmethod
    def report_reference_debug(reference, gaia_config):
        if not gaia_config.get("debug", False):
            return
        score = reference.get("snr_score")
        score_text = "nan" if score is None or not np.isfinite(score) else f"{score:.3g}"
        candidate_scores = []
        for item in reference.get("candidate_summaries", []):
            candidate_score = item.get("snr_score")
            candidate_score = (
                "nan" if candidate_score is None or not np.isfinite(candidate_score)
                else f"{candidate_score:.3g}"
            )
            candidate_scores.append(
                f"{item.get('observatory')}.{item.get('band')}:"
                f"snr={candidate_score}/matches={item.get('matches')}"
            )
        warnings.warn(
            "GAIA reference debug: "
            f"reference={reference.get('observatory')}.{reference.get('band')}, "
            f"wavelength={reference.get('wavelength'):.1f}A, reason={reference.get('reason')}, "
            f"snr={score_text}, detections={len(reference.get('detections', []))}, "
            f"matched={len(reference.get('matched_sources', []))}, "
            f"gaia_fallback={len(reference.get('fallback_sources', []))}, "
            f"candidates=[{'; '.join(candidate_scores)}]",
            UserWarning,
        )

    @staticmethod
    def plot_reference_debug(reference):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        ax.imshow(reference["reference_image"], origin="lower", cmap="bone")
        detections = reference.get("detections", [])
        if detections:
            ax.scatter(
                [item["x"] for item in detections],
                [item["y"] for item in detections],
                s=12,
                facecolors="none",
                edgecolors="gray",
                label="Detected",
            )
        matches = reference.get("matched_sources", [])
        if matches:
            ax.scatter(
                [item["x"] for item in matches],
                [item["y"] for item in matches],
                marker="+",
                color="lime",
                label="GAIA matched",
            )
        fallback_positions = []
        wcs = WCS(reference["reference_header"])
        ny, nx = reference["reference_image"].shape
        for item in reference.get("fallback_sources", []):
            x, y = wcs.all_world2pix(item["ra"], item["dec"], 0)
            if np.isfinite(x) and np.isfinite(y) and 0 <= x < nx and 0 <= y < ny:
                fallback_positions.append((x, y))
        if fallback_positions:
            positions = np.asarray(fallback_positions)
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                marker="x",
                color="orange",
                label="GAIA-only fallback",
            )
        if detections or matches or fallback_positions:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        ax.set_title(
            f"GAIA reference: {reference['observatory']}.{reference['band']} "
            f"({reference['wavelength'] / 10.0:.0f} nm)"
        )
        plt.show()

    @staticmethod
    def plot_mask_debug(mask_image, masked_image, debug_sources):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        masked_sources = debug_sources.get("masked", [])
        rejected_sources = debug_sources.get("rejected", [])
        for plot_image, title in ((mask_image, "Mask image"), (masked_image, "Masked image")):
            _, ax = plt.subplots()
            ax.imshow(plot_image, origin="lower", cmap="bone")

            for index, (x, y, radius) in enumerate(masked_sources):
                ax.add_patch(Circle(
                    (x, y),
                    radius,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.2,
                    label="Masked radius" if index == 0 else None,
                ))
            if masked_sources:
                positions = np.asarray([(x, y) for x, y, _ in masked_sources])
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    marker="+",
                    color="lime",
                    linewidths=1.2,
                    label="Masked center",
                )
            if rejected_sources:
                positions = np.asarray(rejected_sources)
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    marker="x",
                    color="red",
                    linewidths=1.2,
                    label="Rejected center",
                )

            if masked_sources or rejected_sources:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
            ax.set_title(title)
            plt.show()
    
    
    def make_mask(self, image, header, galaxy, psf_fwhm, manual, gaia_mask=None,
                  metadata_resolver=None, debug_sources=None, reference_sources=None,
                  observatory=None, band=None):
        mask, masked, sky = self.py2dmask(
            self,
            image,
            header,
            galaxy,
            psf_fwhm,
            mask_config=gaia_mask,
            metadata_resolver=metadata_resolver,
        )
        segmentation_mask = np.asarray(mask, dtype=bool)
        masks = [segmentation_mask]
        component_counts = {"segmentation": int(np.sum(segmentation_mask))}
        
        if manual is not None:
            manual_mask = self.manual_mask(
                self,
                image,
                header,
                psf_fwhm,
                manual,
                galaxy,
                observatory=observatory,
                band=band,
            )
            masks.append(manual_mask)
            component_counts["manual"] = int(np.sum(manual_mask))
        else:
            component_counts["manual"] = 0

        gaia_source_mask = self.gaia_source_mask(
            self,
            image,
            header,
            psf_fwhm,
            gaia_mask,
            debug_sources=debug_sources,
            reference_sources=reference_sources,
        )
        if gaia_source_mask is not None:
            masks.append(gaia_source_mask)
            component_counts["gaia"] = int(np.sum(gaia_source_mask))
        else:
            component_counts["gaia"] = 0

        combined_mask = np.logical_or.reduce(masks)
        component_counts["combined"] = int(np.sum(combined_mask))
        self.report_mask_component_debug(self, gaia_mask, component_counts)
        masked = np.where(combined_mask, np.nan, image)
            
        return combined_mask, masked, sky
    
    
    @classmethod
    def prepare_manual_mask(cls, manual):
        if manual is None:
            return None
        if not isinstance(manual, pd.DataFrame):
            raise TypeError(
                "manual_mask must be a pandas.DataFrame or None. The legacy "
                "{'coord': ..., 'radius': ...} format is no longer supported; pass a "
                "DataFrame with _RAJ2000, _DEJ2000, radius, and matched_galaxy columns."
            )

        missing = [column for column in cls._MANUAL_MASK_COLUMNS if column not in manual.columns]
        if missing:
            expected = ", ".join(cls._MANUAL_MASK_COLUMNS)
            raise ValueError(
                f"manual_mask is missing required columns: {', '.join(missing)}. "
                f"Expected columns: {expected}."
            )

        prepared = manual.copy(deep=True)
        ra = pd.to_numeric(prepared["_RAJ2000"], errors="coerce").astype(float)
        dec = pd.to_numeric(prepared["_DEJ2000"], errors="coerce").astype(float)
        radius = pd.to_numeric(prepared["radius"], errors="coerce").astype(float)
        galaxy = prepared["matched_galaxy"].astype("string").str.strip()

        finite_coordinates = np.isfinite(ra.to_numpy()) & np.isfinite(dec.to_numpy())
        coordinate_range = (
            finite_coordinates
            & ra.between(0.0, 360.0, inclusive="left").to_numpy()
            & dec.between(-90.0, 90.0, inclusive="both").to_numpy()
        )
        valid_radius = np.isfinite(radius.to_numpy()) & (radius.to_numpy() > 0)
        valid_galaxy = (
            galaxy.notna().to_numpy(dtype=bool)
            & galaxy.ne("").fillna(False).to_numpy(dtype=bool, na_value=False)
        )
        valid = coordinate_range & valid_radius & valid_galaxy

        invalid_count = int(np.sum(~valid))
        if invalid_count:
            nonfinite_coordinates = int(np.sum(~finite_coordinates))
            out_of_range_coordinates = int(np.sum(finite_coordinates & ~coordinate_range))
            invalid_radius = int(np.sum(~valid_radius))
            invalid_galaxy = int(np.sum(~valid_galaxy))
            warnings.warn(
                "Manual mask catalog: skipped "
                f"{invalid_count} invalid row(s) "
                f"(nonfinite_coordinates={nonfinite_coordinates}, "
                f"out_of_range_coordinates={out_of_range_coordinates}, "
                f"invalid_radius={invalid_radius}, invalid_galaxy={invalid_galaxy}).",
                RuntimeWarning,
                stacklevel=2,
            )

        prepared["_RAJ2000"] = ra
        prepared["_DEJ2000"] = dec
        prepared["radius"] = radius
        prepared["matched_galaxy"] = galaxy
        prepared = prepared.loc[valid].copy()
        return prepared

    @staticmethod
    def normalize_manual_galaxy_name(value):
        if value is None or pd.isna(value):
            return None
        normalized = str(value).strip().casefold()
        return normalized or None

    @classmethod
    def manual_sources_for_galaxy(cls, manual, galaxy):
        prepared = cls.prepare_manual_mask(manual)
        galaxy_key = cls.normalize_manual_galaxy_name(galaxy)
        if galaxy_key is None or prepared.empty:
            return prepared.iloc[0:0]

        source_keys = prepared["matched_galaxy"].str.casefold()
        return prepared.loc[source_keys == galaxy_key]

    def manual_mask(self, image, header, psf_fwhm, manual, galaxy, observatory=None, band=None):
        del psf_fwhm  # Manual aperture radii are supplied directly in arcsec.
        sources = self.manual_sources_for_galaxy(manual, galaxy)
        mask = np.zeros(np.shape(image), dtype=bool)
        if sources.empty:
            return mask

        target = "/".join(
            str(value) for value in (galaxy, observatory, band) if value is not None
        )
        try:
            wcs = WCS(header)
        except Exception as exc:
            raise ValueError(f"Could not construct WCS for manual mask target {target}: {exc}") from exc

        pixel_scale = useful_functions.get_pixel_scale(header)
        if not np.isfinite(pixel_scale) or pixel_scale <= 0:
            raise ValueError(
                f"Invalid pixel scale for manual mask target {target}: {pixel_scale}"
            )

        height, width = np.shape(image)
        invalid_projection = 0
        outside_image = 0
        applied = 0

        source_values = sources[["_RAJ2000", "_DEJ2000", "radius"]]
        for ra, dec, radius_arcsec in source_values.itertuples(index=False, name=None):
            radius_pix = float(radius_arcsec) / pixel_scale
            try:
                x, y = wcs.all_world2pix(ra, dec, 0)
            except Exception:
                invalid_projection += 1
                continue

            if not np.all(np.isfinite([x, y, radius_pix])) or radius_pix <= 0:
                invalid_projection += 1
                continue
            if (
                x + radius_pix < 0
                or x - radius_pix >= width
                or y + radius_pix < 0
                or y - radius_pix >= height
            ):
                outside_image += 1
                continue

            aperture_image = CircularAperture(
                positions=(x, y),
                r=radius_pix,
            ).to_mask().to_image(shape=np.shape(image))
            if aperture_image is None:
                outside_image += 1
                continue
            mask |= aperture_image > 0
            applied += 1

        if invalid_projection or outside_image:
            emit_alert(
                f"Manual mask selected={len(sources)}, applied={applied}, "
                f"invalid_projection={invalid_projection}, outside_image={outside_image}.",
                context=target,
                dedupe_key="mask.manual.partially_applied",
            )

        return mask

    def gaia_source_mask(self, image, header, psf_fwhm, gaia_config, debug_sources=None,
                         reference_sources=None):
        if gaia_config is None:
            return None

        masked_sources = None if debug_sources is None else debug_sources.setdefault("masked", [])
        rejected_sources = None if debug_sources is None else debug_sources.setdefault("rejected", [])
        using_reference = reference_sources is not None
        if using_reference:
            catalog = None
            sources = reference_sources
        else:
            catalog = self.load_gaia_catalog(self, image, header, gaia_config)
            sources = self.normalize_gaia_catalog(self, catalog)
        debug_stats = {
            "queried_rows": len(sources) if using_reference else self.catalog_length(self, catalog),
            "normalized_sources": len(sources),
            "reference_sources": int(using_reference),
            "image_candidates": 0,
            "foreground_rejected": 0,
            "disk_protected_skips": 0,
            "radius_failures": 0,
            "fallback_masks": 0,
            "masked_sources": 0,
        }
        if not sources:
            if not using_reference:
                warnings.warn("No source queried from GAIA.", UserWarning)
            self.report_gaia_debug(self, gaia_config, debug_stats)
            return np.zeros_like(image, dtype=bool)

        wcs = WCS(header)
        candidates = []
        ny, nx = image.shape
        if using_reference:
            pixel_scale = useful_functions.get_pixel_scale(header)
            for record in sources:
                source = record.get("source", {}).copy()
                try:
                    if record.get("kind") == "matched":
                        coord = SkyCoord(
                            ra=float(record["ra"]) * u.deg,
                            dec=float(record["dec"]) * u.deg,
                            frame="icrs",
                        )
                    else:
                        coord = self.source_skycoord(source, header=header)
                    x, y = wcs.all_world2pix(coord.ra.deg, coord.dec.deg, 0)
                except Exception:
                    continue
                if np.isfinite(x) and np.isfinite(y) and (0 <= x < nx) and (0 <= y < ny):
                    source["x"] = float(x)
                    source["y"] = float(y)
                    source["reference_kind"] = record.get("kind")
                    reference_radius = record.get("reference_radius_arcsec")
                    source["reference_radius_pix"] = (
                        None if reference_radius is None else float(reference_radius) / pixel_scale
                    )
                    candidates.append(source)
        else:
            for source in sources:
                is_foreground = self.is_gaia_foreground_candidate(self, source, gaia_config)
                if not is_foreground:
                    debug_stats["foreground_rejected"] += 1
                    if rejected_sources is None:
                        continue
                try:
                    x, y = wcs.all_world2pix(source["ra"], source["dec"], 0)
                except Exception:
                    continue
                if not np.isfinite(x) or not np.isfinite(y):
                    continue
                if (0 <= x < nx) and (0 <= y < ny):
                    source = source.copy()
                    source["x"] = float(x)
                    source["y"] = float(y)
                    if is_foreground:
                        candidates.append(source)
                    else:
                        rejected_sources.append((source["x"], source["y"]))

        debug_stats["image_candidates"] = len(candidates)
        if not candidates:
            self.report_gaia_debug(self, gaia_config, debug_stats)
            return np.zeros_like(image, dtype=bool)

        positions = np.array([(source["x"], source["y"]) for source in candidates], dtype=float)
        combined_mask = np.zeros_like(image, dtype=bool)
        pixel_scale = useful_functions.get_pixel_scale(header)
        galaxy_disk = self.galaxy_disk_geometry(self, image, gaia_config)
        for index, source in enumerate(candidates):
            nearest = np.inf
            if len(positions) > 1:
                distances = np.hypot(positions[:, 0] - source["x"], positions[:, 1] - source["y"])
                distances[index] = np.inf
                nearest = np.nanmin(distances)

            radius_mode = "standard"
            inside_disk = self.inside_galaxy_disk(self, source["x"], source["y"], galaxy_disk)
            if inside_disk:
                if not gaia_config.get("mask_bright_stars_on_disk", True):
                    debug_stats["disk_protected_skips"] += 1
                    if rejected_sources is not None:
                        rejected_sources.append((source["x"], source["y"]))
                    continue
                if not self.is_bright_star_on_disk(self, image, source["x"], source["y"], psf_fwhm, gaia_config):
                    debug_stats["disk_protected_skips"] += 1
                    if rejected_sources is not None:
                        rejected_sources.append((source["x"], source["y"]))
                    continue
                radius_mode = gaia_config.get("on_disk_radius_mode", "minimal")

            radius = self.estimate_gaia_mask_radius(
                self,
                image,
                source["x"],
                source["y"],
                psf_fwhm,
                nearest,
                pixel_scale,
                gaia_config,
                radius_mode=radius_mode,
            )
            reference_radius = source.get("reference_radius_pix")
            if reference_radius is not None and np.isfinite(reference_radius) and reference_radius > 0:
                min_radius, max_radius = self.gaia_radius_bounds(
                    self,
                    image,
                    source["x"],
                    source["y"],
                    psf_fwhm,
                    nearest,
                    pixel_scale,
                    gaia_config,
                    radius_mode=radius_mode,
                )
                if max_radius > 0:
                    reference_radius = min(max(float(reference_radius), min_radius), max_radius)
                    radius = reference_radius if radius is None else max(float(radius), reference_radius)
            if radius is None or radius <= 0:
                debug_stats["radius_failures"] += 1
                if inside_disk or not gaia_config.get("mask_all_gaia_outside_disk", True):
                    if rejected_sources is not None:
                        rejected_sources.append((source["x"], source["y"]))
                    continue
                radius = self.gaia_fallback_mask_radius(
                    self,
                    image,
                    source["x"],
                    source["y"],
                    psf_fwhm,
                    nearest,
                    gaia_config,
                )
                if radius is None or radius <= 0:
                    if rejected_sources is not None:
                        rejected_sources.append((source["x"], source["y"]))
                    continue
                debug_stats["fallback_masks"] += 1

            aperture_mask = CircularAperture((source["x"], source["y"]), r=radius).to_mask().to_image(shape=image.shape)
            if aperture_mask is None:
                if rejected_sources is not None:
                    rejected_sources.append((source["x"], source["y"]))
                continue
            combined_mask |= aperture_mask > 0
            debug_stats["masked_sources"] += 1
            if masked_sources is not None:
                masked_sources.append((source["x"], source["y"], float(radius)))

        self.report_gaia_debug(self, gaia_config, debug_stats)
        return combined_mask

    @classmethod
    def _gaia_config(cls, gaia_mask):
        if gaia_mask is None or gaia_mask is False:
            return None

        config = {
            "query": True,
            "catalog": None,
            "local_catalog": None,
            "n_sigma": 2.0,
            "min_radius_fwhm": 1.0,
            "parallax_sigma": 3.0,
            "pm_sigma": 3.0,
            "require_astrometric_foreground": True,
            "error_fill": 999.0,
            "max_radius_fraction": 0.15,
            "bright_sigma": 50.0,
            "saturation_level": None,
            "protect_galaxy_disk": True,
            "disk_protection_scale": 1.0,
            "mask_bright_stars_on_disk": True,
            "on_disk_bright_sigma": 20.0,
            "on_disk_radius_mode": "minimal",
            "on_disk_max_radius_fwhm": 3.0,
            "on_disk_max_radius_arcsec": None,
            "on_disk_n_sigma": 2.0,
            "on_disk_floor_consecutive": 2,
            "on_disk_bright_radius_boost": 1.5,
            "protect_galaxy_segmentation": True,
            "segmentation_protection_scale": 1.0,
            "query_radius_margin": 0.3,
            "gaia_row_limit": -1,
            "mask_all_gaia_outside_disk": True,
            "outside_disk_fallback_radius_fwhm": 1.0,
            "reference_detection_mode": "auto",
            "reference_target": None,
            "reference_optical_range_angstrom": (3500.0, 10000.0),
            "reference_fallback_angstrom": 6000.0,
            "reference_min_matches": 3,
            "reference_min_error_fraction": 0.5,
            "detection_sigma": 5.0,
            "crossmatch_radius_arcsec": None,
            "unmatched_gaia_policy": "significant_or_bright",
            "unmatched_gaia_gmag_max": 15.0,
            "mask_max_wavelength_angstrom": None,
            "debug": False,
            "debug_plot": False,
            "warn": True,
        }

        if isinstance(gaia_mask, dict):
            config.update(gaia_mask)
        elif gaia_mask is not True:
            config["local_catalog"] = gaia_mask

        mode = config.get("reference_detection_mode")
        if mode not in {"auto", "off", "required"}:
            raise ValueError("reference_detection_mode must be 'auto', 'off', or 'required'.")
        error_fraction = float(config.get("reference_min_error_fraction", 0.5))
        if not 0.0 <= error_fraction <= 1.0:
            raise ValueError("reference_min_error_fraction must be between 0 and 1.")
        if int(config.get("reference_min_matches", 3)) < 1:
            raise ValueError("reference_min_matches must be at least 1.")
        if float(config.get("detection_sigma", 5.0)) <= 0:
            raise ValueError("detection_sigma must be positive.")
        match_radius = config.get("crossmatch_radius_arcsec")
        if match_radius is not None and float(match_radius) <= 0:
            raise ValueError("crossmatch_radius_arcsec must be positive.")
        maximum_wavelength = config.get("mask_max_wavelength_angstrom")
        if maximum_wavelength is not None and float(maximum_wavelength) <= 0:
            raise ValueError("mask_max_wavelength_angstrom must be positive.")

        return config

    def load_gaia_catalog(self, image, header, gaia_config, sky_region=None):
        catalog = None
        if gaia_config.get("query", True):
            try:
                if sky_region is None:
                    catalog = self.query_gaia_catalog(self, image, header, gaia_config)
                else:
                    catalog = self.query_gaia_catalog(
                        self,
                        image,
                        header,
                        gaia_config,
                        sky_region=sky_region,
                    )
            except Exception as exc:
                if gaia_config.get("warn", True):
                    warnings.warn(f"GAIA query failed; falling back to local catalog if available: {exc}")
            if self.catalog_length(self, catalog) > 0:
                return catalog

        local_catalog = None
        for key in ("local_catalog", "catalog", "path", "filepath"):
            if gaia_config.get(key) is not None:
                local_catalog = gaia_config.get(key)
                break
        if local_catalog is None:
            return None

        try:
            return self.read_gaia_catalog(self, local_catalog)
        except Exception as exc:
            if gaia_config.get("warn", True):
                warnings.warn(f"Could not read local GAIA catalog; GAIA masking disabled: {exc}")
            return None

    def query_gaia_catalog(self, image, header, gaia_config, sky_region=None):
        center, radius = sky_region or self.image_sky_region(self, image, header, gaia_config)
        from astroquery.gaia import Gaia

        row_limit = gaia_config.get("gaia_row_limit", -1)
        original_row_limit = getattr(Gaia, "ROW_LIMIT", None)
        if row_limit is not None:
            Gaia.ROW_LIMIT = row_limit
        try:
            job = Gaia.cone_search_async(center, radius=radius)
            return job.get_results()
        finally:
            if row_limit is not None:
                Gaia.ROW_LIMIT = original_row_limit

    def image_sky_region(self, image, header, gaia_config=None):
        wcs = WCS(header)
        ny, nx = image.shape
        center = wcs.pixel_to_world((nx - 1) / 2.0, (ny - 1) / 2.0)
        if not isinstance(center, SkyCoord):
            center = SkyCoord(center)

        corners = np.array(
            [
                [0, 0],
                [nx - 1, 0],
                [0, ny - 1],
                [nx - 1, ny - 1],
            ],
            dtype=float,
        )
        corner_sky = wcs.pixel_to_world(corners[:, 0], corners[:, 1])
        margin = 1.05 if gaia_config is None else gaia_config.get("query_radius_margin", 1.05)
        radius = np.nanmax(center.separation(corner_sky)) * float(margin)
        if not np.isfinite(radius.to_value(u.deg)) or radius <= 0 * u.deg:
            radius = (
                max(image.shape)
                * useful_functions.get_pixel_scale(header)
                / 3600.0
                * float(margin)
            ) * u.deg
        return center, radius

    def read_gaia_catalog(self, catalog):
        if isinstance(catalog, (str, Path)):
            import pandas as pd

            path = Path(catalog).expanduser()
            try:
                return pd.read_csv(path, sep=None, engine="python", keep_default_na=False)
            except Exception:
                return pd.read_csv(path, sep=r"\s+", engine="python", keep_default_na=False)
        return catalog

    def catalog_length(self, catalog):
        if catalog is None:
            return 0
        try:
            return len(catalog)
        except Exception:
            return 0

    def normalize_gaia_catalog(self, catalog):
        if self.catalog_length(self, catalog) == 0:
            return []

        try:
            import pandas as pd

            if isinstance(catalog, pd.DataFrame):
                dataframe = catalog
            elif hasattr(catalog, "to_pandas"):
                dataframe = catalog.to_pandas()
            elif isinstance(catalog, dict):
                dataframe = pd.DataFrame(catalog)
            else:
                dataframe = pd.DataFrame(catalog)
        except Exception:
            return []

        column_lookup = {str(column).lower(): column for column in dataframe.columns}
        sources = []
        for catalog_index, (_, row) in enumerate(dataframe.iterrows()):
            ra = self.row_value(self, row, column_lookup, self._RA_COLUMNS)
            dec = self.row_value(self, row, column_lookup, self._DEC_COLUMNS)
            ra = self.to_float(self, ra)
            dec = self.to_float(self, dec)
            if not np.isfinite(ra) or not np.isfinite(dec):
                continue

            sources.append(
                {
                    "catalog_index": catalog_index,
                    "source_id": self.row_value(self, row, column_lookup, self._SOURCE_ID_COLUMNS),
                    "ref_epoch": self.to_float(
                        self,
                        self.row_value(self, row, column_lookup, self._REF_EPOCH_COLUMNS),
                    ),
                    "ra": ra,
                    "dec": dec,
                    "gmag": self.to_float(self, self.row_value(self, row, column_lookup, self._GMAG_COLUMNS)),
                    "plx": self.to_float(self, self.row_value(self, row, column_lookup, self._PLX_COLUMNS)),
                    "e_plx": self.to_float(self, self.row_value(self, row, column_lookup, self._E_PLX_COLUMNS)),
                    "pm": self.to_float(self, self.row_value(self, row, column_lookup, self._PM_COLUMNS)),
                    "e_pm": self.to_float(self, self.row_value(self, row, column_lookup, self._E_PM_COLUMNS)),
                    "pmra": self.to_float(self, self.row_value(self, row, column_lookup, self._PMRA_COLUMNS)),
                    "e_pmra": self.to_float(self, self.row_value(self, row, column_lookup, self._E_PMRA_COLUMNS)),
                    "pmdec": self.to_float(self, self.row_value(self, row, column_lookup, self._PMDEC_COLUMNS)),
                    "e_pmdec": self.to_float(self, self.row_value(self, row, column_lookup, self._E_PMDEC_COLUMNS)),
                    "qso": self.row_value(self, row, column_lookup, self._QSO_COLUMNS),
                    "gal": self.row_value(self, row, column_lookup, self._GAL_COLUMNS),
                    "ruwe": self.to_float(self, self.row_value(self, row, column_lookup, self._RUWE_COLUMNS)),
                }
            )
        return sources

    def row_value(self, row, column_lookup, candidates):
        for candidate in candidates:
            column = column_lookup.get(candidate.lower())
            if column is not None:
                return row[column]
        return None

    def to_float(self, value):
        if value is None:
            return np.nan
        try:
            if isinstance(value, str) and not value.strip():
                return np.nan
            return float(value)
        except Exception:
            return np.nan

    def is_gaia_foreground_candidate(self, source, gaia_config):
        if self.flag_true(self, source.get("qso")) or self.flag_true(self, source.get("gal")):
            return False

        has_astrometric_test = False
        plx = source.get("plx", np.nan)
        e_plx = source.get("e_plx", np.nan)
        if np.isfinite(plx) and np.isfinite(e_plx) and e_plx > 0:
            has_astrometric_test = True
            if abs(plx / e_plx) >= gaia_config.get("parallax_sigma", 3.0):
                return True

        pm_sig = self.proper_motion_significance(self, source)
        if np.isfinite(pm_sig):
            has_astrometric_test = True
            if pm_sig >= gaia_config.get("pm_sigma", 3.0):
                return True

        if gaia_config.get("require_astrometric_foreground", False) and has_astrometric_test:
            return False
        return True

    def proper_motion_significance(self, source):
        pm = source.get("pm", np.nan)
        e_pm = source.get("e_pm", np.nan)
        if np.isfinite(pm) and np.isfinite(e_pm) and e_pm > 0:
            return abs(pm / e_pm)

        pmra = source.get("pmra", np.nan)
        e_pmra = source.get("e_pmra", np.nan)
        pmdec = source.get("pmdec", np.nan)
        e_pmdec = source.get("e_pmdec", np.nan)
        components = []
        if np.isfinite(pmra) and np.isfinite(e_pmra) and e_pmra > 0:
            components.append((pmra / e_pmra) ** 2)
        if np.isfinite(pmdec) and np.isfinite(e_pmdec) and e_pmdec > 0:
            components.append((pmdec / e_pmdec) ** 2)
        if not components:
            return np.nan
        return np.sqrt(np.sum(components))

    def flag_true(self, value):
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"", "nan", "none", "null", "--"}:
                return False
            return stripped in {"1", "true", "t", "yes", "y"}
        try:
            if not np.isfinite(value):
                return False
        except (TypeError, ValueError):
            pass
        try:
            return bool(int(value))
        except Exception:
            return bool(value)

    def report_gaia_debug(self, gaia_config, debug_stats):
        if not gaia_config.get("debug", False):
            return

        summary = ", ".join(f"{key}={value}" for key, value in debug_stats.items())
        warnings.warn(f"GAIA mask debug: {summary}", UserWarning)

    def report_mask_component_debug(self, mask_config, component_counts):
        if mask_config is None or not mask_config.get("debug", False):
            return

        summary = ", ".join(f"{key}={value}" for key, value in component_counts.items())
        warnings.warn(f"Mask component debug: {summary}", UserWarning)

    def galaxy_disk_geometry(self, image, gaia_config):
        if not gaia_config.get("protect_galaxy_disk", True):
            return None

        first_error = None
        try:
            x0, y0, a, b, theta = useful_functions.get_galaxy_radius(image)
            x0, y0, a, b, theta = map(float, (x0, y0, a, b, theta))
            if not all(np.isfinite(value) for value in (x0, y0, a, b, theta)):
                raise ValueError("non-finite disk geometry")
            if a <= 0 or b <= 0:
                raise ValueError("non-positive disk radius")
            ny, nx = image.shape
            if a >= nx and b >= ny:
                raise ValueError("galaxy disk segmentation returned the full image")
        except Exception as exc:
            first_error = exc
            try:
                x0, y0, a, b, theta = self.segmented_galaxy_disk_geometry(self, image)
            except Exception as fallback_exc:
                if gaia_config.get("warn", True):
                    warnings.warn(
                        "Could not estimate galaxy disk; GAIA disk protection skipped: "
                        f"{first_error}; fallback failed: {fallback_exc}"
                    )
                return None

        scale = float(gaia_config.get("disk_protection_scale", 1.0))
        return {
            "x0": x0,
            "y0": y0,
            "a": a * scale,
            "b": b * scale,
            "theta": theta,
        }

    def segmented_galaxy_disk_geometry(self, image):
        finite_image = image[np.isfinite(image)]
        if finite_image.size == 0:
            raise ValueError("no finite pixels")

        _, median, std = sigma_clipped_stats(finite_image, sigma=3.0)
        if not np.isfinite(std) or std <= 0:
            std = np.nanstd(finite_image)
        if not np.isfinite(std) or std <= 0:
            raise ValueError("no usable image contrast")

        segment_map = detect_sources(image, threshold=median + 3.0 * std, npixels=5)
        if segment_map is None:
            raise ValueError("no segmented galaxy disk")

        labels, counts = np.unique(segment_map.data[segment_map.data > 0], return_counts=True)
        if labels.size == 0:
            raise ValueError("no positive segmentation labels")

        label = labels[np.argmax(counts)]
        disk_mask = segment_map.data == label
        yy, xx = np.nonzero(disk_mask)
        if xx.size < 5:
            raise ValueError("segmented disk has too few pixels")

        values = np.asarray(image[disk_mask], dtype=float)
        weights = values - median
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        if np.sum(weights) <= 0:
            weights = np.ones_like(values, dtype=float)

        x0 = float(np.average(xx, weights=weights))
        y0 = float(np.average(yy, weights=weights))
        dx = xx - x0
        dy = yy - y0
        cov_xx = float(np.average(dx * dx, weights=weights))
        cov_yy = float(np.average(dy * dy, weights=weights))
        cov_xy = float(np.average(dx * dy, weights=weights))
        cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvec = eigvecs[:, order[0]]
        if eigvals[0] <= 0 or eigvals[1] <= 0:
            raise ValueError("invalid segmented disk covariance")

        a = float(2.0 * np.sqrt(eigvals[0]))
        b = float(2.0 * np.sqrt(eigvals[1]))
        theta = float(math.atan2(eigvec[1], eigvec[0]))
        ny, nx = image.shape
        if a >= nx and b >= ny:
            raise ValueError("segmented disk returned the full image")

        return x0, y0, a, b, theta

    def galaxy_segmentation_geometry(self, image, mask_config):
        if mask_config is None or not mask_config.get("protect_galaxy_segmentation", True):
            return None

        segmentation_config = dict(mask_config)
        segmentation_config["protect_galaxy_disk"] = True
        segmentation_config["disk_protection_scale"] = mask_config.get("segmentation_protection_scale", 1.0)
        return self.galaxy_disk_geometry(self, image, segmentation_config)

    def galaxy_ellipse_mask(self, image_shape, galaxy_disk):
        if galaxy_disk is None:
            return np.zeros(image_shape, dtype=bool)

        yy, xx = np.indices(image_shape)
        dx = xx - galaxy_disk["x0"]
        dy = yy - galaxy_disk["y0"]
        cos_t = math.cos(galaxy_disk["theta"])
        sin_t = math.sin(galaxy_disk["theta"])
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        value = (x_rot / galaxy_disk["a"]) ** 2 + (y_rot / galaxy_disk["b"]) ** 2
        return np.isfinite(value) & (value <= 1.0)

    def inside_galaxy_disk(self, x, y, galaxy_disk):
        if galaxy_disk is None:
            return False

        dx = x - galaxy_disk["x0"]
        dy = y - galaxy_disk["y0"]
        cos_t = math.cos(galaxy_disk["theta"])
        sin_t = math.sin(galaxy_disk["theta"])
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        value = (x_rot / galaxy_disk["a"]) ** 2 + (y_rot / galaxy_disk["b"]) ** 2
        return np.isfinite(value) and value <= 1.0

    def is_bright_star_on_disk(self, image, x, y, psf_fwhm, gaia_config):
        stats = self.gaia_source_local_stats(self, image, x, y, psf_fwhm)
        if stats is None:
            return False

        saturation_level = gaia_config.get("saturation_level")
        saturated = saturation_level is not None and stats["peak"] >= float(saturation_level)
        bright = stats["peak"] > stats["median"] + gaia_config.get("on_disk_bright_sigma", 20.0) * stats["std"]
        return saturated or bright

    def gaia_source_local_stats(self, image, x, y, psf_fwhm):
        ny, nx = image.shape
        radius = max(3.0, 4.0 * float(psf_fwhm))
        radius = min(radius, x + 1, y + 1, nx - x, ny - y)
        if radius < 1:
            return None

        radius_int = int(np.ceil(radius))
        x0 = max(0, int(np.floor(x - radius_int)))
        x1 = min(nx, int(np.ceil(x + radius_int + 1)))
        y0 = max(0, int(np.floor(y - radius_int)))
        y1 = min(ny, int(np.ceil(y + radius_int + 1)))
        cutout = image[y0:y1, x0:x1]
        if cutout.size == 0:
            return None

        yy, xx = np.indices(cutout.shape)
        rr = np.hypot(xx + x0 - x, yy + y0 - y)
        finite = np.isfinite(cutout)
        annulus_inner = max(2.0 * float(psf_fwhm), float(psf_fwhm) + 2.0)
        annulus = finite & (rr >= annulus_inner) & (rr <= radius)
        if np.sum(annulus) >= 10:
            _, local_median, local_std = sigma_clipped_stats(cutout[annulus], sigma=3.0)
        else:
            finite_image = image[np.isfinite(image)]
            if finite_image.size == 0:
                return None
            _, local_median, local_std = sigma_clipped_stats(finite_image, sigma=3.0)

        if not np.isfinite(local_std) or local_std <= 0:
            local_std = np.finfo(float).eps

        core = finite & (rr <= max(1.0, 0.5 * float(psf_fwhm)))
        if np.sum(core) == 0:
            return None

        peak = np.nanmax(cutout[core])
        if not np.isfinite(peak) or not np.isfinite(local_median):
            return None
        return {"peak": peak, "median": local_median, "std": local_std}

    def gaia_radius_bounds(self, image, x, y, psf_fwhm, nearest_source_distance, pixel_scale,
                           gaia_config, radius_mode="standard"):
        ny, nx = image.shape
        min_radius = max(1.0, float(psf_fwhm) * gaia_config.get("min_radius_fwhm", 1.0))
        max_radius = gaia_config.get("max_radius_pix")
        if max_radius is None:
            max_radius = max(3.0 * min_radius, min(image.shape) * gaia_config.get("max_radius_fraction", 0.15))

        max_radius_arcsec = gaia_config.get("max_radius_arcsec")
        if max_radius_arcsec is not None:
            max_radius = min(max_radius, float(max_radius_arcsec) / pixel_scale)

        if radius_mode == "minimal":
            on_disk_max_fwhm = gaia_config.get("on_disk_max_radius_fwhm")
            if on_disk_max_fwhm is not None:
                max_radius = min(max_radius, float(psf_fwhm) * float(on_disk_max_fwhm))
            on_disk_max_arcsec = gaia_config.get("on_disk_max_radius_arcsec")
            if on_disk_max_arcsec is not None:
                max_radius = min(max_radius, float(on_disk_max_arcsec) / pixel_scale)
            max_radius = max(max_radius, min_radius)

        if np.isfinite(nearest_source_distance):
            max_radius = min(max_radius, 0.45 * nearest_source_distance)

        edge_cap = min(x + 1, y + 1, nx - x, ny - y)
        max_radius = min(max_radius, edge_cap)
        return min_radius, max_radius

    def gaia_fallback_mask_radius(self, image, x, y, psf_fwhm, nearest_source_distance, gaia_config):
        ny, nx = image.shape
        radius = max(1.0, float(psf_fwhm) * gaia_config.get("outside_disk_fallback_radius_fwhm", 1.0))
        if np.isfinite(nearest_source_distance):
            radius = min(radius, 0.45 * nearest_source_distance)

        edge_cap = min(x + 1, y + 1, nx - x, ny - y)
        radius = min(radius, edge_cap)
        if radius <= 0:
            return None
        return radius

    def estimate_gaia_mask_radius(self, image, x, y, psf_fwhm, nearest_source_distance, pixel_scale,
                                  gaia_config, radius_mode="standard"):
        min_radius, max_radius = self.gaia_radius_bounds(
            self,
            image,
            x,
            y,
            psf_fwhm,
            nearest_source_distance,
            pixel_scale,
            gaia_config,
            radius_mode=radius_mode,
        )
        if max_radius <= 0:
            return None

        radius_int = int(np.ceil(max_radius))
        ny, nx = image.shape
        x0 = max(0, int(np.floor(x - radius_int)))
        x1 = min(nx, int(np.ceil(x + radius_int + 1)))
        y0 = max(0, int(np.floor(y - radius_int)))
        y1 = min(ny, int(np.ceil(y + radius_int + 1)))
        cutout = image[y0:y1, x0:x1]
        if cutout.size == 0:
            return None

        yy, xx = np.indices(cutout.shape)
        rr = np.hypot(xx + x0 - x, yy + y0 - y)
        finite = np.isfinite(cutout)
        annulus_inner = max(2.0 * min_radius, min_radius + 2.0)
        annulus = finite & (rr >= annulus_inner) & (rr <= max_radius)
        if np.sum(annulus) >= 10:
            _, local_median, local_std = sigma_clipped_stats(cutout[annulus], sigma=3.0)
        else:
            _, local_median, local_std = sigma_clipped_stats(image[np.isfinite(image)], sigma=3.0)
        if not np.isfinite(local_std) or local_std <= 0:
            local_std = np.finfo(float).eps

        core = finite & (rr <= max(1.0, 0.5 * min_radius))
        if np.sum(core) == 0:
            return None
        peak = np.nanmax(cutout[core])
        threshold_sigma = gaia_config.get("n_sigma", 2.0)
        if radius_mode == "minimal":
            threshold_sigma = gaia_config.get("on_disk_n_sigma", threshold_sigma)
        threshold = local_median + threshold_sigma * local_std
        if not np.isfinite(peak) or peak <= threshold:
            return None

        last_excess_radius = 0.0
        first_floor_radius = None
        floor_count = 0
        floor_required = max(1, int(gaia_config.get("on_disk_floor_consecutive", 2)))
        for radius in range(1, int(np.floor(max_radius)) + 1):
            annulus_mask = finite & (rr >= radius - 0.5) & (rr < radius + 0.5)
            if np.sum(annulus_mask) == 0:
                continue
            annulus_value = np.nanmedian(cutout[annulus_mask])
            if np.isfinite(annulus_value) and annulus_value > threshold:
                last_excess_radius = float(radius)
                floor_count = 0
            elif radius_mode == "minimal" and radius >= min_radius:
                floor_count += 1
                if floor_count >= floor_required:
                    first_floor_radius = float(radius)
                    break

        if radius_mode == "minimal" and first_floor_radius is not None:
            radius = max(min_radius, first_floor_radius)
        else:
            radius = max(min_radius, last_excess_radius + 1.0)
        saturation_level = gaia_config.get("saturation_level")
        bright_sigma = gaia_config.get("bright_sigma", 50.0)
        saturated = saturation_level is not None and peak >= float(saturation_level)
        very_bright = peak > local_median + bright_sigma * local_std
        if radius_mode == "minimal":
            radius = min(radius * gaia_config.get("on_disk_bright_radius_boost", 1.5), max_radius)
        elif saturated or very_bright:
            radius = max(radius, min(2.0 * radius, max_radius))

        return min(radius, max_radius)

    def py2dmask(self, image, header, galaxy, psf_fwhm, mask_config=None, metadata_resolver=None):
        if mask_config is None:
            mask_config = self._gaia_config(True)

        center_coord = useful_functions.get_sky_loc(
            galaxy,
            header=header,
            metadata_resolver=metadata_resolver,
            required=True,
            image_shape=image.shape,
        )
        
        wcs = WCS(header)
        x, y = wcs.all_world2pix(center_coord.ra.deg, center_coord.dec.deg, 0)
        
        from astropy.stats import sigma_clipped_stats
        mean, median, std = sigma_clipped_stats(image, sigma=10.0, mask=np.where(image == 0, True, False))
        
        from photutils.segmentation import detect_sources
        segment_map = detect_sources(image, threshold=5 * std, npixels=30)
        if segment_map is None:
            masked_image = image
            return np.zeros_like(masked_image), masked_image, np.full_like(masked_image, fill_value=median)

        segmentation_data = np.array(segment_map.data, copy=True)
        protected_labels = set()
        try:
            center_label = int(segmentation_data[int(y), int(x)])
            if center_label > 0:
                protected_labels.add(center_label)
        except Exception:
            pass

        galaxy_geometry = self.galaxy_segmentation_geometry(self, image, mask_config)
        galaxy_mask = self.galaxy_ellipse_mask(self, image.shape, galaxy_geometry)
        if np.any(galaxy_mask):
            labels = np.unique(segmentation_data[galaxy_mask & (segmentation_data > 0)])
            protected_labels.update(int(label) for label in labels)

        if protected_labels:
            segmentation_data[np.isin(segmentation_data, list(protected_labels))] = 0

        mask = self.circular_segmentation_mask(segmentation_data, psf_fwhm)
        masked_image = np.where(mask, np.nan, image) if np.any(mask) else image
        return mask, masked_image, np.full_like(masked_image, fill_value=median)

    @staticmethod
    def circular_segmentation_mask(segmentation_data, psf_fwhm):
        """Replace each labeled segmentation component with an equal-area circle."""
        segmentation_data = np.asarray(segmentation_data)
        circular_mask = np.zeros(segmentation_data.shape, dtype=bool)
        try:
            minimum_radius = float(psf_fwhm)
        except (TypeError, ValueError):
            minimum_radius = 0.0
        if not np.isfinite(minimum_radius) or minimum_radius < 0:
            minimum_radius = 0.0

        for label, slices in enumerate(find_objects(segmentation_data), start=1):
            if slices is None:
                continue
            component = segmentation_data[slices] == label
            area = int(np.count_nonzero(component))
            if area == 0:
                continue

            local_y, local_x = np.nonzero(component)
            center_x = float(np.mean(local_x) + slices[1].start)
            center_y = float(np.mean(local_y) + slices[0].start)
            radius = max(minimum_radius, math.sqrt(area / math.pi))
            aperture_mask = CircularAperture(
                (center_x, center_y),
                r=radius,
            ).to_mask(method="center")
            large_slices, small_slices = aperture_mask.get_overlap_slices(
                segmentation_data.shape
            )
            if large_slices is None:
                continue
            circular_mask[large_slices] |= aperture_mask.data[small_slices] > 0

        return circular_mask
    

    def own_mask(self, image, header, galaxy, psf_fwhm):
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        
        daofind = DAOStarFinder(fwhm=psf_fwhm, threshold=15.*std)
        sources = daofind(image - median)
        
        if sources is None:
            emit_alert(
                "No sources were found for legacy PSF masking.",
                context=galaxy,
                dedupe_key="mask.legacy.no_sources",
            )
            return image, image, image

        self.x0, self.y0, self.a, self.b, self.theta = useful_functions.get_galaxy_radius(
            image,
            context=galaxy,
        )

        dist = self.is_point_in_rotated_ellipse(self, sources['xcentroid'], sources['ycentroid'])

        good = dist > 1
        filtered_sources = sources[good]
        if not filtered_sources.indices:
            return image, image, image

        psf_model = MoffatPSF()
        psf_model.alpha.fixed = False
        psf_model.flux.fixed = False
        fit_shape =(int(psf_fwhm*2.0) * 2 + 1, int(psf_fwhm*2.0) * 2 + 1)
        bkgstat = MMMBackground()
        localbkg_estimator = LocalBackground(5, 10, bkgstat)
        
        psfphot = PSFPhotometry(psf_model, fit_shape)
        phot = psfphot(image - median, init_params=filtered_sources["xcentroid", "ycentroid", "flux"],
                       localbkg_estimator=localbkg_estimator)
        
        if phot is None:
            emit_alert(
                "No PSF photometry result was produced by legacy masking.",
                context=galaxy,
                dedupe_key="mask.legacy.no_photometry",
            )
            return image, image, image
        
        emit_detail(f"Phot: {phot}")
        
        resid = psfphot.make_residual_image(image - median)
        mask_image = np.where(image - median - resid > 0, 1, 0)
        masked_image = resid.copy()
        
        return mask_image, masked_image, np.full_like(masked_image, fill_value=median)


    def __make_mask(self, image, header, galaxy, psf_fwhm, metadata_resolver=None):
        psf_fwhm = np.abs(psf_fwhm)
        center_coord = useful_functions.get_sky_loc(
            galaxy,
            header=header,
            metadata_resolver=metadata_resolver,
            required=True,
            image_shape=image.shape,
        )
        image_x, image_y = image.shape
        box_shape = [int(image_x/5), int(image_y/5)]
        filter_shape = [int(psf_fwhm*2.0) * 2 + 1, int(psf_fwhm*2.0) * 2 + 1]
        
        wcs = WCS(header)
        x, y = wcs.all_world2pix(center_coord.ra.deg, center_coord.dec.deg, 0)

        bkg_estimator = MedianBackground()
        try:    
            bkg = Background2D(image, box_shape, filter_size=filter_shape, bkg_estimator=bkg_estimator)
        except ValueError:
            emit_alert(
                "Legacy masking background failed; retrying with a smaller box.",
                context=galaxy,
                dedupe_key="mask.legacy.smaller_background",
            )
            bkg = Background2D(image, (int(box_shape[0]/5), int(box_shape[1]/5)), filter_size=filter_shape, bkg_estimator=bkg_estimator)
        threshold = 1.5*bkg.background_rms

        segment_map = detect_sources(image, threshold, npixels=5)
        if segment_map == None:
            return image, image, image

        sky_map = np.nonzero(segment_map.data)
        sky_image = image.copy()
        sky_image[sky_map] = np.nan

        label_main = segment_map.data[int(y), int(x)]
        if label_main != 0:
            segment_map.remove_labels([label_main])

        mask = np.nonzero(segment_map.data)
        masked_image = image.copy()
        masked_image[mask] = np.nan
            
        mask_image = np.zeros_like(image)
        mask_image[mask] = image[mask]
        
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        daofind = DAOStarFinder(fwhm=psf_fwhm, threshold=15.*std)
        sources = daofind(masked_image - mean)
        
        if sources is None:
            emit_alert(
                "No sources were found for legacy PSF masking.",
                context=galaxy,
                dedupe_key="mask.legacy.no_sources",
            )
            return masked_image, masked_image, masked_image

        self.x0, self.y0, self.a, self.b, self.theta = useful_functions.get_galaxy_radius(
            image,
            context=galaxy,
        )

        dist = self.is_point_in_rotated_ellipse(self, sources['xcentroid'], sources['ycentroid'])

        good = dist > 1
        filtered_sources = sources[good]
        if not filtered_sources.indices:
            return image, image, image

        psf_model = MoffatPSF()
        psf_model.alpha.fixed = False
        psf_model.flux.fixed = False
        fit_shape = (31, 31) 
        psfphot = PSFPhotometry(psf_model, fit_shape,
                                aperture_radius=psf_fwhm*5)
        phot = psfphot(masked_image - mean, init_params=filtered_sources["xcentroid", "ycentroid", "flux"])
        if phot is None:
            emit_alert(
                "No PSF photometry result was produced by legacy masking.",
                context=galaxy,
                dedupe_key="mask.legacy.no_photometry",
            )
            return masked_image, masked_image, masked_image
        
        resid = psfphot.make_residual_image(masked_image - mean)
        mask_image = np.where(masked_image - mean - resid > 0, 1, 0)
        masked_image = resid.copy()
        
        return mask_image, masked_image, sky_image
        

    def is_point_in_rotated_ellipse(self, x, y):

        dx = x - self.x0
        dy = y - self.y0

        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        x_rot =  dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        value = (x_rot / self.a)**2 + (y_rot / self.b)**2
        return value
