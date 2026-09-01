import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from Spec7DT.manipulation.mask import Masking
from Spec7DT.reduction.registration import Register


def make_wcs_header(galaxy="NGC1", pixel_scale_arcsec=1.0):
    header = fits.Header()
    header["OBJECT"] = galaxy
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRPIX1"] = 5.5
    header["CRPIX2"] = 5.5
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    header["CDELT1"] = -pixel_scale_arcsec / 3600.0
    header["CDELT2"] = pixel_scale_arcsec / 3600.0
    return header


def make_projection_header(projection="TAN", size=21, matrix="pc"):
    header = make_wcs_header(pixel_scale_arcsec=1.0)
    header["CRPIX1"] = (size + 1) / 2
    header["CRPIX2"] = (size + 1) / 2
    header["CTYPE1"] = f"RA---{projection}"
    header["CTYPE2"] = f"DEC--{projection}"

    if matrix == "cd":
        del header["CDELT1"]
        del header["CDELT2"]
        header["CD1_1"] = -1.0 / 3600.0
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 1.0 / 3600.0
    else:
        header["PC1_1"] = 1.0
        header["PC1_2"] = 0.0
        header["PC2_1"] = 0.0
        header["PC2_2"] = 1.0

    if projection == "TPV":
        header["PV1_0"] = 0.0
        header["PV1_1"] = 1.0
        header["PV2_0"] = 0.0
        header["PV2_1"] = 1.0
    elif projection == "ZPN":
        header["PROJP1"] = 1.0
        header["PROJP3"] = -45.0
        header["PROJP5"] = 0.0
    elif projection == "TAN-SIP":
        header["A_ORDER"] = 2
        header["B_ORDER"] = 2
        header["A_2_0"] = 1.0e-8
        header["B_0_2"] = -1.0e-8
    return header


class FixedResolver:
    def get_skycoord(self, *_args, **_kwargs):
        return SkyCoord(10.0 * u.deg, 20.0 * u.deg)


class SyntheticImageSet:
    def __init__(self):
        self.data = {}
        self.header = {}
        self.error = {}
        self.files = {}

    def add_target(
        self,
        root,
        galaxy,
        observatory,
        band,
        value,
        error_value=0.1,
        pixel_scale_arcsec=1.0,
    ):
        header = make_wcs_header(galaxy, pixel_scale_arcsec=pixel_scale_arcsec)
        data = np.full((10, 10), value, dtype=np.float32)
        error = np.full((10, 10), error_value, dtype=np.float32)
        source_path = Path(root) / f"{galaxy}-{observatory}-{band}.fits"
        fits.writeto(source_path, data, header, overwrite=True)

        for tree in (self.data, self.header, self.error, self.files):
            tree.setdefault(galaxy, {}).setdefault(observatory, {})
        self.data[galaxy][observatory][band] = data
        self.header[galaxy][observatory][band] = header
        self.error[galaxy][observatory][band] = error
        self.files[galaxy][observatory][band] = str(source_path)

    def update_data(self, image_data, galaxy_name, observatory, band):
        self.data[galaxy_name][observatory][band] = image_data

    def update_header(self, updated_header, galaxy_name, observatory, band):
        self.header[galaxy_name][observatory][band] = updated_header

    def update_error(self, error_data, galaxy_name, observatory, band):
        self.error[galaxy_name][observatory][band] = error_data


class FakeSwarp:
    def __init__(self, write_outputs=True, offset=10.0):
        self.write_outputs = write_outputs
        self.offset = offset
        self.calls = []

    def __call__(
        self,
        input_list,
        output=None,
        dump_dir=None,
        resample_dir=None,
        log_file=None,
    ):
        run_dir = Path(dump_dir)
        manifest_path = Path(input_list)
        resample_dir = Path(resample_dir)
        entries = manifest_path.read_text(encoding="utf-8").splitlines()
        self.calls.append({"run_dir": run_dir, "entries": entries})
        Path(log_file).write_text("fake swarp\n" + "\n".join(entries), encoding="utf-8")

        if self.write_outputs:
            for entry in entries:
                input_path = run_dir / entry
                data = fits.getdata(input_path).astype(np.float32)
                header = fits.getheader(input_path)
                header["TESTREG"] = True
                output_path = resample_dir / f"{input_path.stem}.resamp{input_path.suffix}"
                fits.writeto(output_path, data + self.offset, header, overwrite=True)

        return "fake swarp"


class TestRegisterImageSet(unittest.TestCase):
    def test_manifest_contains_only_current_image_set_and_success_is_cleaned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_set = SyntheticImageSet()
            image_set.add_target(root, "NGC1", "7DT", "m600", 1.0)
            image_set.add_target(root, "NGC1", "7DT", "m650", 2.0)

            legacy_dir = root / "SWarp"
            legacy_dir.mkdir()
            stale_path = legacy_dir / "unpro_NGC1-SPIRE-PMW.fits"
            fits.writeto(stale_path, np.zeros((3, 3), dtype=np.float32), overwrite=True)

            fake_swarp = FakeSwarp()
            with mock.patch.object(Register, "swarp", side_effect=fake_swarp):
                run_dirs = Register.register_image_set(image_set)

            self.assertEqual(len(fake_swarp.calls), 1)
            entries = fake_swarp.calls[0]["entries"]
            self.assertEqual(len(entries), 4)
            self.assertTrue(any("m600" in entry for entry in entries))
            self.assertTrue(any("m650" in entry for entry in entries))
            self.assertFalse(any("SPIRE" in entry or "PMW" in entry for entry in entries))
            self.assertTrue(stale_path.is_file())

            run_dir = run_dirs["NGC1"]
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {"NGC1.list", "swarp.log"},
            )
            self.assertEqual(
                (run_dir / "NGC1.list").read_text(encoding="utf-8").splitlines(),
                entries,
            )
            self.assertEqual(image_set.data["NGC1"]["7DT"]["m600"].dtype, np.float32)
            np.testing.assert_allclose(image_set.data["NGC1"]["7DT"]["m600"], 11.0)
            np.testing.assert_allclose(image_set.error["NGC1"]["7DT"]["m650"], 10.1)
            self.assertTrue(image_set.header["NGC1"]["7DT"]["m600"]["TESTREG"])

    def test_registration_reports_each_registered_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_set = SyntheticImageSet()
            image_set.add_target(root, "NGC1", "7DT", "m600", 1.0)
            image_set.add_target(root, "NGC1", "7DT", "m650", 2.0)
            fake_swarp = FakeSwarp()
            reporter = mock.Mock()

            with mock.patch.object(Register, "swarp", side_effect=fake_swarp):
                Register.register_image_set(image_set, reporter=reporter)

        reporter.set_target.assert_called_once_with("NGC1 (SWarp: 2 images)")
        self.assertEqual(
            reporter.advance_target.call_args_list,
            [
                mock.call("NGC1/7DT/m600"),
                mock.call("NGC1/7DT/m650"),
            ],
        )

    def test_repeated_runs_use_distinct_workspaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_set = SyntheticImageSet()
            image_set.add_target(root, "NGC1", "7DT", "m600", 1.0)
            fake_swarp = FakeSwarp()

            with mock.patch.object(Register, "swarp", side_effect=fake_swarp):
                first = Register.register_image_set(image_set)["NGC1"]
                second = Register.register_image_set(image_set)["NGC1"]

            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertEqual(len(fake_swarp.calls), 2)
            self.assertNotEqual(
                fake_swarp.calls[0]["run_dir"],
                fake_swarp.calls[1]["run_dir"],
            )

    def test_swarp_runs_once_per_galaxy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_set = SyntheticImageSet()
            image_set.add_target(root, "NGC1", "7DT", "m600", 1.0)
            image_set.add_target(root, "NGC2", "7DT", "m650", 2.0)
            fake_swarp = FakeSwarp()

            with mock.patch.object(Register, "swarp", side_effect=fake_swarp):
                run_dirs = Register.register_image_set(image_set)

            self.assertEqual(set(run_dirs), {"NGC1", "NGC2"})
            self.assertEqual(len(fake_swarp.calls), 2)
            self.assertTrue(all(len(call["entries"]) == 2 for call in fake_swarp.calls))
            np.testing.assert_allclose(image_set.data["NGC1"]["7DT"]["m600"], 11.0)
            np.testing.assert_allclose(image_set.data["NGC2"]["7DT"]["m650"], 12.0)

    def test_missing_outputs_preserve_workspace_and_original_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_set = SyntheticImageSet()
            image_set.add_target(root, "NGC1", "7DT", "m600", 1.0)
            original = image_set.data["NGC1"]["7DT"]["m600"].copy()
            fake_swarp = FakeSwarp(write_outputs=False)

            with mock.patch.object(Register, "swarp", side_effect=fake_swarp):
                with self.assertRaisesRegex(RuntimeError, "Workspace retained"):
                    Register.register_image_set(image_set)

            run_dir = fake_swarp.calls[0]["run_dir"]
            self.assertTrue((run_dir / "inputs").is_dir())
            self.assertTrue((run_dir / "resampled").is_dir())
            self.assertTrue((run_dir / "NGC1.list").is_file())
            self.assertTrue((run_dir / "swarp.log").is_file())
            np.testing.assert_array_equal(
                image_set.data["NGC1"]["7DT"]["m600"],
                original,
            )

    def test_image_set_pipeline_step_runs_once_and_in_order(self):
        from Spec7DT.utils.pipeline import ImageProcessingPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_set = SyntheticImageSet()
            image_set.add_target(root, "NGC1", "7DT", "m600", 1.0)
            image_set.add_target(root, "NGC1", "7DT", "m650", 2.0)
            events = []

            def before_step(band):
                events.append(f"before:{band}")

            def set_step(image_set):
                events.append(f"set:{sum(len(bands) for obs in image_set.data.values() for bands in obs.values())}")

            def after_step(band):
                events.append(f"after:{band}")

            pipeline = ImageProcessingPipeline(image_set, metadata_resolver=object())
            pipeline.add_step(before_step, step_name="Before")
            pipeline.add_image_set_step(set_step, step_name="Set")
            pipeline.add_step(after_step, step_name="After")
            pipeline.execute(band_filter=["m600"])

            self.assertEqual(
                events,
                ["before:m600", "set:2", "after:m600"],
            )


class TestWCSNormalization(unittest.TestCase):
    def test_wcs_classifier_preserves_similarly_named_science_metadata(self):
        header = make_projection_header("TAN")
        header["PROJP3"] = -50.0
        header["PS1_0"] = "projection parameter"
        header["PSCALET1"] = 1.25
        header["PSIGDET"] = 7.5
        header["BUNIT"] = "MJy/sr"

        cleaned = Register._without_wcs_cards(header)

        self.assertNotIn("PROJP3", cleaned)
        self.assertNotIn("PS1_0", cleaned)
        self.assertEqual(cleaned["PSCALET1"], 1.25)
        self.assertEqual(cleaned["PSIGDET"], 7.5)
        self.assertEqual(cleaned["BUNIT"], "MJy/sr")

    def test_replacement_removes_stale_primary_alternate_and_distortion_wcs(self):
        base = make_projection_header("ZPN", size=71)
        base["BUNIT"] = "Jy/pixel"
        base["PSCALET1"] = 10.0
        base["CTYPE1A"] = "RA---ARC"
        base["CTYPE2A"] = "DEC--ARC"
        base["CRPIX1A"] = 36.0
        base["CRPIX2A"] = 36.0
        base["CRVAL1A"] = 10.0
        base["CRVAL2A"] = 20.0
        base["CDELT1A"] = -1.0 / 3600.0
        base["CDELT2A"] = 1.0 / 3600.0
        base["A_ORDER"] = 2
        base["A_2_0"] = 1.0e-8
        base["CPDIS1"] = "LOOKUP"
        base["DP1.AXIS.1"] = 1
        base["D2IM1A"] = "LOOKUP"
        base["NAXIS3"] = 4

        authoritative = make_projection_header("TAN", size=71, matrix="cd")
        authoritative["TESTREG"] = True
        normalized = Register._replace_celestial_wcs(
            base,
            authoritative,
            shape=(71, 71),
            context="synthetic registration",
        )

        for key in (
            "PROJP1", "PROJP3", "PROJP5", "CTYPE1A", "CTYPE2A",
            "CRPIX1A", "CRVAL1A", "A_ORDER", "A_2_0", "CPDIS1",
            "DP1.AXIS.1", "D2IM1A", "NAXIS3",
        ):
            self.assertNotIn(key, normalized)
        self.assertEqual(normalized["CTYPE1"], "RA---TAN")
        self.assertEqual(normalized["CTYPE2"], "DEC--TAN")
        self.assertEqual(normalized["BUNIT"], "Jy/pixel")
        self.assertEqual(normalized["PSCALET1"], 10.0)
        self.assertTrue(normalized["TESTREG"])
        self.assertEqual((normalized["NAXIS2"], normalized["NAXIS1"]), (71, 71))

    def test_registered_then_trimmed_wcs_keeps_target_finite_and_inside(self):
        target = SkyCoord(10.0 * u.deg, 20.0 * u.deg)
        original = make_projection_header("ZPN", size=71)
        registered = make_projection_header("TAN", size=71)
        merged = Register._merge_registered_header(
            original,
            registered,
            shape=(71, 71),
            context="mixed projection registration",
        )

        image = np.ones((71, 71), dtype=np.float32)
        error = np.full_like(image, 0.1)
        cut_image, cut_header, cut_error = Register.trim_sky(
            image,
            merged,
            error,
            target,
            (50, 50),
        )

        x, y = WCS(cut_header).celestial.world_to_pixel(target)
        self.assertTrue(np.isfinite(x) and np.isfinite(y))
        self.assertTrue(0 <= x < 50 and 0 <= y < 50)
        self.assertEqual(cut_image.shape, (50, 50))
        self.assertEqual(cut_error.shape, cut_image.shape)
        self.assertEqual((cut_header["NAXIS2"], cut_header["NAXIS1"]), cut_image.shape)
        self.assertFalse(any(key.startswith("PROJP") for key in cut_header))

    def test_supported_projection_and_matrix_forms_round_trip(self):
        cases = (
            ("TAN", "pc"),
            ("TAN", "cd"),
            ("ARC", "pc"),
            ("TPV", "pc"),
            ("ZPN", "pc"),
            ("TAN-SIP", "pc"),
        )
        target = SkyCoord(10.0 * u.deg, 20.0 * u.deg)

        for projection, matrix in cases:
            with self.subTest(projection=projection, matrix=matrix):
                authoritative = make_projection_header(projection, matrix=matrix)
                normalized = Register._replace_celestial_wcs(
                    fits.Header({"OBJECT": "NGC1", "BUNIT": "Jy"}),
                    authoritative,
                    shape=(21, 21),
                    context=f"{projection}/{matrix}",
                    target_coord=target,
                )
                x, y = WCS(normalized).celestial.world_to_pixel(target)
                self.assertTrue(np.isfinite(x) and np.isfinite(y))
                self.assertAlmostEqual(x, 10.0, places=5)
                self.assertAlmostEqual(y, 10.0, places=5)
                self.assertEqual(normalized["BUNIT"], "Jy")

    def test_invalid_wcs_shape_and_target_fail_with_context(self):
        valid = make_projection_header("TAN")
        no_celestial = fits.Header({"NAXIS": 2, "NAXIS1": 21, "NAXIS2": 21})

        with self.assertRaisesRegex(ValueError, "missing celestial.*two celestial"):
            Register._replace_celestial_wcs(
                fits.Header(),
                no_celestial,
                shape=(21, 21),
                context="missing celestial",
            )
        with self.assertRaisesRegex(ValueError, "cube target.*positive 2-D"):
            Register._replace_celestial_wcs(
                fits.Header(),
                valid,
                shape=(2, 21, 21),
                context="cube target",
            )
        with self.assertRaisesRegex(ValueError, "off-image target.*outside image"):
            Register._replace_celestial_wcs(
                fits.Header(),
                valid,
                shape=(21, 21),
                context="off-image target",
                target_coord=SkyCoord(30.0 * u.deg, 40.0 * u.deg),
            )

    def test_mixed_projection_registration_trim_and_mask_sequence(self):
        projections = ("TAN", "ARC", "TPV", "ZPN", "TAN-SIP")
        resolver = FixedResolver()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_set = SyntheticImageSet()
            image_set.psf = {"NGC1": {}}
            for index, projection in enumerate(projections):
                observatory = f"OBS{index}"
                image_set.add_target(root, "NGC1", observatory, "band", 0.0)
                header = make_projection_header(projection, size=21)
                data = np.zeros((21, 21), dtype=np.float32)
                error = np.full_like(data, 0.1)
                image_set.header["NGC1"][observatory]["band"] = header
                image_set.data["NGC1"][observatory]["band"] = data
                image_set.error["NGC1"][observatory]["band"] = error
                image_set.psf["NGC1"].setdefault(observatory, {})["band"] = 2.0

            fake_swarp = FakeSwarp(offset=0.0)
            with mock.patch.object(Register, "swarp", side_effect=fake_swarp):
                Register.register_image_set(image_set)

            for index, _projection in enumerate(projections):
                observatory = f"OBS{index}"
                Register.trim(
                    image_set.data["NGC1"][observatory]["band"],
                    image_set.header["NGC1"][observatory]["band"],
                    image_set.error["NGC1"][observatory]["band"],
                    "NGC1",
                    observatory,
                    "band",
                    image_set,
                    trim_size=(15, 15),
                    metadata_resolver=resolver,
                )
                mask, masked, _ = Masking.py2dmask(
                    Masking,
                    image_set.data["NGC1"][observatory]["band"],
                    image_set.header["NGC1"][observatory]["band"],
                    "NGC1",
                    2.0,
                    mask_config=False,
                    metadata_resolver=resolver,
                )
                self.assertEqual(mask.shape, (15, 15))
                self.assertEqual(masked.shape, (15, 15))


class TestSwarpCommand(unittest.TestCase):
    def test_swarp_uses_argv_without_shell_or_center(self):
        class SuccessfulProcess:
            def __init__(self):
                self.stdout = iter(["SWarp output\n"])
                self.returncode = 0

            def wait(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            manifest = run_dir / "inputs.list"
            manifest.write_text("inputs/example.fits\n", encoding="utf-8")
            log_path = run_dir / "swarp.log"

            with mock.patch(
                "Spec7DT.reduction.registration.subprocess.Popen",
                return_value=SuccessfulProcess(),
            ) as popen:
                Register.swarp(
                    input_list=manifest,
                    output=run_dir / "coadd.fits",
                    dump_dir=run_dir,
                    resample_dir=run_dir / "resampled",
                    log_file=log_path,
                )

            command = popen.call_args.args[0]
            kwargs = popen.call_args.kwargs
            self.assertIsInstance(command, list)
            self.assertEqual(command[0], "swarp")
            self.assertNotIn("-CENTER", command)
            self.assertNotIn("-CENTER_TYPE", command)
            self.assertIs(kwargs["shell"], False)
            self.assertEqual(kwargs["cwd"], str(run_dir))
            self.assertIn(str(run_dir / "coadd.fits"), command)
            self.assertIn(str(run_dir / "resampled"), command)
            self.assertIn("SWarp output", log_path.read_text(encoding="utf-8"))

    def test_nonzero_exit_preserves_log(self):
        class FailedProcess:
            def __init__(self):
                self.stdout = iter(["fatal error\n"])
                self.returncode = 1

            def wait(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            manifest = run_dir / "inputs.list"
            manifest.write_text("inputs/example.fits\n", encoding="utf-8")
            log_path = run_dir / "swarp.log"

            with mock.patch(
                "Spec7DT.reduction.registration.subprocess.Popen",
                return_value=FailedProcess(),
            ):
                with self.assertRaisesRegex(RuntimeError, "return code 1"):
                    Register.swarp(
                        input_list=manifest,
                        dump_dir=run_dir,
                        log_file=log_path,
                    )

            self.assertIn("fatal error", log_path.read_text(encoding="utf-8"))


@unittest.skipUnless(shutil.which("swarp"), "SWarp executable is not installed")
class TestSwarpIntegration(unittest.TestCase):
    def test_real_swarp_ignores_stale_legacy_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_set = SyntheticImageSet()
            image_set.add_target(root, "NGC1", "7DT", "m600", 1.0)
            image_set.add_target(root, "NGC1", "7DT", "m650", 2.0)

            legacy_dir = root / "SWarp"
            legacy_dir.mkdir()
            stale_header = make_wcs_header("NGC1", pixel_scale_arcsec=10.0)
            fits.writeto(
                legacy_dir / "unpro_NGC1-SPIRE-PMW.fits",
                np.ones((10, 10), dtype=np.float32),
                stale_header,
                overwrite=True,
            )

            run_dir = Register.register_image_set(image_set)["NGC1"]

            manifest = (run_dir / "NGC1.list").read_text(encoding="utf-8")
            log = (run_dir / "swarp.log").read_text(encoding="utf-8")
            self.assertEqual(len(manifest.splitlines()), 4)
            self.assertNotIn("SPIRE", manifest)
            self.assertNotIn("PMW", log)
            self.assertNotIn("-CENTER", log.splitlines()[0])
            self.assertEqual(
                image_set.data["NGC1"]["7DT"]["m600"].shape,
                image_set.data["NGC1"]["7DT"]["m650"].shape,
            )
            scales = proj_plane_pixel_scales(
                WCS(image_set.header["NGC1"]["7DT"]["m600"])
            ) * 3600.0
            np.testing.assert_allclose(np.abs(scales), 1.0, rtol=0.01)
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {"NGC1.list", "swarp.log"},
            )


if __name__ == "__main__":
    unittest.main()
