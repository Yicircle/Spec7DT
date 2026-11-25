import inspect
from pathlib import Path
from time import perf_counter
from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import asdict, dataclass, field, replace

from ..plot.plot import DrawGalaxy
from ..handlers.filter_handler import Filters
from ..core import AnalysisProduct, CatalogFrame
from .checkpoint import CheckpointManager
from .metadata import GalaxyMetadataConfig, GalaxyMetadataResolver
from .reporting import RichPipelineReporter, pipeline_warning_context, reporter_context


@dataclass(frozen=True)
class PipelineOutputConfig:
    progress: bool = True
    show_alerts: bool = True
    live_alerts: bool = True
    pdf_path: str | Path | None = None
    pdf_dpi: int = 300
    show_plots: bool | None = None
    overwrite: bool = False
    save_checkpoints: bool = False
    checkpoint_dir: str | Path | None = None

    @classmethod
    def from_value(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("output_config must be a PipelineOutputConfig, dict, or None")

    def __post_init__(self):
        if (
            not isinstance(self.pdf_dpi, int)
            or isinstance(self.pdf_dpi, bool)
            or self.pdf_dpi <= 0
        ):
            raise ValueError("pdf_dpi must be a positive integer")
        if self.show_plots is not None and not isinstance(self.show_plots, bool):
            raise TypeError("show_plots must be bool or None")
        if not isinstance(self.show_alerts, bool):
            raise TypeError("show_alerts must be bool")
        if not isinstance(self.live_alerts, bool):
            raise TypeError("live_alerts must be bool")
        if not isinstance(self.save_checkpoints, bool):
            raise TypeError("save_checkpoints must be bool")
        if self.checkpoint_dir is not None and not isinstance(
            self.checkpoint_dir, (str, Path)
        ):
            raise TypeError("checkpoint_dir must be a path-like string, Path, or None")
        if self.checkpoint_dir is not None and not self.save_checkpoints:
            raise ValueError("checkpoint_dir requires save_checkpoints=True")

    @property
    def resolved_show_plots(self):
        if self.show_plots is not None:
            return self.show_plots
        return self.pdf_path is None


class ImageProcessingPipeline:
    def __init__(self, galaxy_image_set, metadata_resolver=None, filter_config=None):
        self.galaxy_image_set = galaxy_image_set
        self.metadata_resolver = metadata_resolver or GalaxyMetadataResolver()
        self.filter_config = filter_config or {}
        self.pipeline_steps = []
        self.step_configs = {}
    
    def add_step(
        self,
        function,
        config=None,
        step_name=None,
        *,
        step_id=None,
        mutates=(),
        product_kind=None,
    ):
        """파이프라인에 처리 단계 추가"""
        if step_name is None:
            step_name = function.__name__
        
        self.pipeline_steps.append({
            'id': step_id or function.__name__,
            'name': step_name,
            'function': function,
            'config': config or {},
            'scope': 'image',
            'mutates': tuple(mutates),
            'product_kind': product_kind,
        })

    def add_image_set_step(
        self,
        function,
        config=None,
        step_name=None,
        *,
        step_id=None,
        mutates=(),
        product_kind=None,
    ):
        """Add a step that runs once against the complete GalaxyImageSet."""
        if step_name is None:
            step_name = function.__name__

        self.pipeline_steps.append({
            'id': step_id or function.__name__,
            'name': step_name,
            'function': function,
            'config': config or {},
            'scope': 'image_set',
            'mutates': tuple(mutates),
            'product_kind': product_kind,
        })
        
    def remove_step(self, step_name):
        """특정 단계 제거"""
        self.pipeline_steps = [step for step in self.pipeline_steps 
                              if step['name'] != step_name]
    
    def execute(
        self,
        galaxy_filter=None,
        observatory_filter=None,
        band_filter=None,
        plot_step=False,
        verbose=False,
        output_config=None,
        reporter=None,
        checkpoint_manager=None,
        completed_step_ids=None,
    ):
        """파이프라인 실행"""
        output_config = PipelineOutputConfig.from_value(output_config)
        if checkpoint_manager is not None and any(
            value is not None
            for value in (galaxy_filter, observatory_filter, band_filter)
        ):
            raise ValueError("Checkpointed execution does not support target filters")
        if output_config.pdf_path is not None and not isinstance(plot_step, dict):
            raise ValueError("pdf_path requires plot_step to be a selector dictionary")

        pdf_path = None
        pdf_pages = None
        if output_config.pdf_path is not None:
            pdf_path = Path(output_config.pdf_path).expanduser()
            if pdf_path.exists() and not output_config.overwrite:
                raise FileExistsError(
                    f"Plot PDF already exists: {pdf_path}. Set overwrite=True to replace it."
                )
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_pages = PdfPages(
                pdf_path,
                metadata={"Title": "Spec7DT pipeline steps", "Creator": "Spec7DT"},
            )

        show_plots = output_config.resolved_show_plots
        reporter = reporter or RichPipelineReporter(
            progress=output_config.progress,
            verbose=verbose,
            show_alerts=output_config.show_alerts,
            live_alerts=output_config.live_alerts,
        )
        timings = []
        active_step = None
        active_target = None
        succeeded = False
        completed_step_ids = set(completed_step_ids or ())
        remaining_steps = [
            step for step in self.pipeline_steps if step["id"] not in completed_step_ids
        ]
        reporter.start(len(remaining_steps))

        with reporter_context(reporter), pipeline_warning_context(reporter):
            try:
                for step_index, step in enumerate(self.pipeline_steps, start=1):
                    if step["id"] in completed_step_ids:
                        continue
                    active_step = step["name"]
                    active_target = None
                    step_products = []
                    if step.get("scope") == "image_set":
                        total_targets = self._image_count()
                    else:
                        total_targets = self._count_filtered_targets(
                            galaxy_filter,
                            observatory_filter,
                            band_filter,
                        )
                    if total_targets == 0:
                        raise ValueError(
                            f"No images available for pipeline step '{step['name']}'. "
                            "Check the GalaxyImageSet contents and filename parsing."
                        )

                    reporter.start_step(step["name"], total_targets)
                    step_start = perf_counter()
                    processed_targets = 0

                    if step.get("scope") == "image_set":
                        active_target = "GalaxyImageSet"
                        reporter.set_target(step["name"], active_target)
                        before = reporter.step_completed
                        delegated_progress, product = self._execute_image_set_step(
                            step,
                            verbose,
                            reporter=reporter,
                        )
                        if isinstance(product, AnalysisProduct):
                            step_products.append({"product": product})
                        if not delegated_progress or reporter.step_completed == before:
                            reporter.advance_target(advance=total_targets)
                        processed_targets = total_targets
                    else:
                        for galaxy, observatory, band, image, header, error in self._get_filtered_targets(
                            galaxy_filter,
                            observatory_filter,
                            band_filter,
                        ):
                            active_target = f"{galaxy}/{observatory}/{band}"
                            reporter.set_target(step["name"], active_target)
                            elapsed, product = self._execute_step(
                                step,
                                galaxy,
                                observatory,
                                band,
                                image,
                                header,
                                error,
                                verbose,
                            )
                            if isinstance(product, AnalysisProduct):
                                step_products.append({
                                    "galaxy": galaxy,
                                    "observatory": observatory,
                                    "band": band,
                                    "product": product,
                                })
                            reporter.detail(
                                f"✓ {step['name']} completed for {active_target} "
                                f"in {elapsed:.3f}s"
                            )
                            reporter.advance_target()
                            processed_targets += 1

                    if checkpoint_manager is not None:
                        checkpoint_manager.complete_step(
                            step_index,
                            step,
                            products=step_products,
                        )

                    step_elapsed = perf_counter() - step_start
                    timings.append({
                        "step": step["name"],
                        "targets": processed_targets,
                        "elapsed": step_elapsed,
                    })
                    reporter.finish_step(step["name"])

                    if isinstance(plot_step, dict):
                        figure = None
                        try:
                            plot_kwargs = {
                                "galaxy": plot_step.get("galaxy"),
                                "obs": plot_step.get("obs"),
                                "band": plot_step.get("band"),
                                "step": step["name"],
                            }
                            if not show_plots:
                                plot_kwargs["show"] = False
                            plot_result = DrawGalaxy.plot_step(
                                self.galaxy_image_set,
                                **plot_kwargs,
                            )
                            figure = plot_result[0]
                            if pdf_pages is not None:
                                pdf_pages.savefig(
                                    figure,
                                    dpi=output_config.pdf_dpi,
                                    bbox_inches="tight",
                                )
                        finally:
                            if figure is not None and not show_plots:
                                plt.close(figure)
                succeeded = True
            except Exception as exc:
                reporter.failure(active_step or "Pipeline", active_target, exc)
                raise
            finally:
                if pdf_pages is not None:
                    pdf_pages.close()
                reporter.stop()

        if succeeded:
            summary_kwargs = {"pdf_path": pdf_path}
            if checkpoint_manager is not None:
                summary_kwargs["checkpoint_dirs"] = checkpoint_manager.output_dirs
            reporter.summary(timings, **summary_kwargs)
        return self.galaxy_image_set

    def _image_count(self):
        return sum(
            len(bands)
            for observatories in self.galaxy_image_set.data.values()
            for bands in observatories.values()
        )

    def _count_filtered_targets(self, galaxy_filter, observatory_filter, band_filter):
        count = 0
        for galaxy, observatories in self.galaxy_image_set.data.items():
            if galaxy_filter and galaxy not in galaxy_filter:
                continue
            for observatory, bands in observatories.items():
                if observatory_filter and observatory not in observatory_filter:
                    continue
                for band in bands:
                    if band_filter and band not in band_filter:
                        continue
                    count += 1
        return count

    def _last_target(self):
        last_target = None
        for galaxy, observatories in self.galaxy_image_set.data.items():
            for observatory, bands in observatories.items():
                for band in bands:
                    last_target = (galaxy, observatory, band)
        return last_target
    
    def _get_filtered_targets(self, galaxy_filter, observatory_filter, band_filter):
        """필터 조건에 맞는 (galaxy, observatory, band) 조합 반환"""
        for galaxy in self.galaxy_image_set.data:
            if galaxy_filter and galaxy not in galaxy_filter:
                continue
                
            for observatory in self.galaxy_image_set.data[galaxy]:
                if observatory_filter and observatory not in observatory_filter:
                    continue
                    
                for band in self.galaxy_image_set.data[galaxy][observatory]:
                    if band_filter and band not in band_filter:
                        continue
                    
                    target_image = self.galaxy_image_set.data[galaxy][observatory][band]
                    target_header = self.galaxy_image_set.header[galaxy][observatory][band]
                    try:
                        target_error = self.galaxy_image_set.error[galaxy][observatory][band]
                    except (AttributeError, KeyError, TypeError):
                        target_error = None
                    
                    yield (galaxy, observatory, band, target_image, target_header, target_error)
    
    def _execute_step(self, step, galaxy, observatory, band, image, header, error, verbose):
        """Execute each step"""
        function = step['function']
        config = step['config']
        
        sig = inspect.signature(function)
        
        # image_data, header, error_data, galaxy_name, observatory, band, image_set
        kwargs = {'image_set': self.galaxy_image_set,
                  'image_data':image, 'header':header, 'error_data':error,
                  'galaxy_name': galaxy, 'observatory': observatory, 'band': band,
                  'metadata_resolver': self.metadata_resolver,
                  'filter_config': self.filter_config}
        
        kwargs.update(config)
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        target_start = perf_counter()
        result = function(**filtered_kwargs)
        if (
            result is not None
            and step.get("product_kind")
            and not isinstance(result, AnalysisProduct)
        ):
            result = AnalysisProduct(kind=step["product_kind"], data=result)
        elapsed = perf_counter() - target_start
        return elapsed, result

    def _execute_image_set_step(self, step, verbose, reporter=None):
        """Execute a step once for the complete, unfiltered image set."""
        function = step['function']
        config = step['config']
        sig = inspect.signature(function)
        kwargs = {
            'image_set': self.galaxy_image_set,
            'metadata_resolver': self.metadata_resolver,
            'filter_config': self.filter_config,
            'reporter': reporter,
        }
        kwargs.update(config)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        target_start = perf_counter()
        result = function(**filtered_kwargs)
        if (
            result is not None
            and step.get("product_kind")
            and not isinstance(result, AnalysisProduct)
        ):
            result = AnalysisProduct(kind=step["product_kind"], data=result)
        elapsed = perf_counter() - target_start
        if reporter is not None:
            reporter.detail(
                f"✓ {step['name']} completed for GalaxyImageSet in {elapsed:.3f}s"
            )
        return "reporter" in sig.parameters, result


from ..reduction.registration import Register
from .unit import conversion
from ..reduction.background import (
    BackgroundConfig,
    backgroundSubtraction,
    estimateMissingError,
    has_missing_error_maps,
    residualBackgroundSubtraction,
)
from ..reduction.PSF import (
    PSFConvolutionConfig,
    PSFConvolutionEngine,
    PointSpreadFunction,
)
from ..reduction.PSF_safe import SafePSFConfig, SafePointSpreadFunction
from ..manipulation.reddening import Reddening
from ..manipulation.mask import Masking
from ..manipulation.sky_interpolate import interpolate_sky
from ..division.binning import Bin
from ..division.cutout import CutRegion

from .file_generator import inputGenerator

@dataclass
class PipelineConfig:
    background: bool = False
    psf: bool = True
    psfconv: bool = True
    register: bool = True
    trim: bool = True
    unit: bool = True
    dered: bool = True
    mask: bool = True
    skyinter: bool = True
    bin: bool = True
    cutout: bool = True
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_processes(cls, processes=None):
        if isinstance(processes, cls):
            return processes
        if not processes:
            return cls()
        known = {field_name for field_name in cls.__dataclass_fields__ if field_name != "extras"}
        values = {key: bool(value) for key, value in processes.items() if key in known}
        extras = {key: value for key, value in processes.items() if key not in known}
        return cls(**values, extras=extras)

    def enabled(self, name):
        if hasattr(self, name):
            return bool(getattr(self, name))
        return bool(self.extras.get(name, False))


def execute_pipeline(
    galaxy_image_set=None,
    cat_type=None,
    processes=None,
    plot_step=False,
    verbose=False,
    trim_size=None,
    manual_mask: pd.DataFrame | None = None,
    gaia_mask=None,
    bin=1,
    box_size=None,
    cut_coeff=1.5,
    config: PipelineConfig = None,
    galaxy_metadata: dict = None,
    metadata_config: GalaxyMetadataConfig | dict = None,
    filter_config: dict = None,
    psfconv_config: PSFConvolutionConfig | dict = None,
    psf_strategy: Literal["safe", "legacy"] | None = None,
    safe_psf_config: SafePSFConfig | dict | None = None,
    background_config: BackgroundConfig | dict = None,
    output_config: PipelineOutputConfig | dict = None,
    resume_from: str | Path | None = None,
):
    checkpoint_manager = None
    completed_step_ids = ()
    resume_settings = None
    if resume_from is not None:
        if galaxy_image_set is not None:
            raise ValueError("galaxy_image_set and resume_from cannot be used together")
        conflicting = []
        for name, value, default in (
            ("processes", processes, None),
            ("trim_size", trim_size, None),
            ("manual_mask", manual_mask, None),
            ("gaia_mask", gaia_mask, None),
            ("bin", bin, 1),
            ("box_size", box_size, None),
            ("cut_coeff", cut_coeff, 1.5),
            ("config", config, None),
            ("galaxy_metadata", galaxy_metadata, None),
            ("metadata_config", metadata_config, None),
            ("filter_config", filter_config, None),
            ("psfconv_config", psfconv_config, None),
            ("psf_strategy", psf_strategy, None),
            ("safe_psf_config", safe_psf_config, None),
            ("background_config", background_config, None),
        ):
            differs = value is not None if default is None else value != default
            if differs:
                conflicting.append(name)
        if conflicting:
            raise ValueError(
                "Resume uses the science settings stored in the checkpoint; remove: "
                + ", ".join(conflicting)
            )

        resumed = CheckpointManager.resume(resume_from)
        galaxy_image_set = resumed.image_set
        checkpoint_manager = resumed.manager
        completed_step_ids = resumed.completed_step_ids
        resume_settings = resumed.settings
        stored_cat_type = resume_settings["cat_type"]
        if cat_type is not None and cat_type != stored_cat_type:
            raise ValueError(
                f"cat_type {cat_type!r} does not match checkpoint value {stored_cat_type!r}"
            )
        cat_type = stored_cat_type

        pipeline_values = dict(resume_settings["pipeline_config"])
        pipeline_extras = dict(pipeline_values.pop("extras", {}))
        pipeline_config = PipelineConfig(**pipeline_values, extras=pipeline_extras)
        trim_size = resume_settings.get("trim_size")
        manual_mask = resume_settings.get("manual_mask")
        gaia_mask = resume_settings.get("gaia_mask")
        bin = resume_settings.get("bin", 1)
        box_size = resume_settings.get("box_size")
        cut_coeff = resume_settings.get("cut_coeff", 1.5)
        galaxy_metadata = resume_settings.get("galaxy_metadata")
        metadata_config = resume_settings.get("metadata_config")
        filter_config = resume_settings.get("filter_config")
        psfconv_config = resume_settings.get("psfconv_config")
        psf_strategy = resume_settings.get("psf_strategy", "legacy")
        safe_psf_config = resume_settings.get("safe_psf_config")
        background_config = resume_settings.get("background_config")
    else:
        if galaxy_image_set is None:
            raise TypeError("galaxy_image_set is required unless resume_from is provided")
        if cat_type is None:
            raise TypeError("cat_type is required unless it is restored from a checkpoint")
        pipeline_config = PipelineConfig.from_processes(
            config if config is not None else processes
        )

    if resume_settings is not None and (
        background_config is None
        or (isinstance(background_config, dict) and "mode" not in background_config)
    ):
        # Manifests written before automatic background selection must retain
        # their original step plan and native-background behavior.
        background_config = {
            **(background_config or {}),
            "mode": "legacy",
            "post_residual": False,
        }

    if psf_strategy is None:
        psf_strategy = "legacy" if psfconv_config is not None else "safe"
    psf_strategy = str(psf_strategy).lower()
    if psf_strategy not in {"safe", "legacy"}:
        raise ValueError("psf_strategy must be 'safe' or 'legacy'")
    if psf_strategy == "safe" and psfconv_config is not None:
        raise ValueError(
            "psfconv_config belongs to the legacy PSF strategy; use safe_psf_config "
            "or set psf_strategy='legacy'"
        )
    if psf_strategy == "legacy" and safe_psf_config is not None:
        raise ValueError("safe_psf_config requires psf_strategy='safe'")
    psfconv_config = (
        PSFConvolutionConfig.from_value(psfconv_config)
        if psf_strategy == "legacy"
        else None
    )
    safe_psf_config = (
        SafePSFConfig.from_value(safe_psf_config)
        if psf_strategy == "safe"
        else None
    )
    background_config = BackgroundConfig.from_value(background_config)
    output_config = PipelineOutputConfig.from_value(output_config)
    if output_config.pdf_path is not None and not isinstance(plot_step, dict):
        raise ValueError("pdf_path requires plot_step to be a selector dictionary")
    selected_pretrim_fov = (
        safe_psf_config.pretrim_fov
        if psf_strategy == "safe"
        else psfconv_config.pretrim_fov
    )
    if not pipeline_config.enabled("psfconv") and selected_pretrim_fov is not None:
        raise ValueError("pretrim_fov requires the psfconv pipeline step to be enabled")
    manual_mask = Masking.prepare_manual_mask(manual_mask)
    if isinstance(metadata_config, dict):
        metadata_config = GalaxyMetadataConfig(**metadata_config)
    metadata_resolver = GalaxyMetadataResolver(metadata=galaxy_metadata, config=metadata_config)

    if (
        psf_strategy == "safe"
        and pipeline_config.enabled("psf")
        and safe_psf_config.mosaic_synthesis is not None
        and "psf_extract" not in completed_step_ids
    ):
        from ..handlers.spherex_mosaic_psf import (
            build_spherex_mosaic_psf_manifest,
            merge_psf_manifests,
            spherex_mosaic_synthesis_required,
            synthesis_config_for_analysis_footprint,
        )

        synthesis_config = synthesis_config_for_analysis_footprint(
            galaxy_image_set,
            safe_psf_config.mosaic_synthesis,
            pretrim_fov=safe_psf_config.pretrim_fov,
            trim_size=trim_size if pipeline_config.enabled("trim") else None,
            metadata_resolver=metadata_resolver,
        )
        synthesis_config = replace(
            synthesis_config,
            progress=bool(synthesis_config.progress and output_config.progress),
        )
        safe_psf_config = replace(
            safe_psf_config,
            mosaic_synthesis=synthesis_config,
        )
        if spherex_mosaic_synthesis_required(
            galaxy_image_set,
            safe_psf_config.psf_manifest,
            synthesis_config.tile_bounds,
        ):
            build_result = build_spherex_mosaic_psf_manifest(
                galaxy_image_set,
                synthesis_config,
                calibration_registry=safe_psf_config.calibration_registry,
                existing_manifest=safe_psf_config.psf_manifest,
            )
            safe_psf_config = replace(
                safe_psf_config,
                psf_manifest=merge_psf_manifests(
                    safe_psf_config.psf_manifest,
                    build_result.manifest_path,
                ),
            )
            builds = list(getattr(galaxy_image_set, "_spherex_psf_builds", ()))
            builds.append(build_result)
            galaxy_image_set._spherex_psf_builds = builds

    filter_options = {"allow_svo": True, "cache": True, "warn": True, "unknown_policy": "best_effort"}
    if filter_config:
        filter_options.update(filter_config)

    gaia_mask_enabled = pipeline_config.enabled("mask") and gaia_mask is not None and gaia_mask is not False
    if pipeline_config.enabled("dered") or pipeline_config.enabled("psf") or gaia_mask_enabled:
        Filters.ensure_filters_for_image_set(galaxy_image_set, **filter_options)
    if gaia_mask_enabled and not hasattr(galaxy_image_set, "_gaia_mask_reference_cache"):
        setattr(galaxy_image_set, "_gaia_mask_reference_cache", {})
    
    
    pipeline1 = ImageProcessingPipeline(
        galaxy_image_set,
        metadata_resolver=metadata_resolver,
        filter_config=filter_options,
    )

    if pipeline_config.enabled("background"):
        pipeline1.add_step(
            backgroundSubtraction,
            config={"background_config": background_config},
            step_id="background",
            mutates=(
                ("data", "error")
                if background_config.legacy_only
                else ("data", "error", "header")
            ),
        )
    elif has_missing_error_maps(galaxy_image_set):
        pipeline1.add_step(
            estimateMissingError,
            config={"background_config": background_config},
            step_name="Estimate Missing Error",
            step_id="estimate_error",
            mutates=("error",),
        )
    if pipeline_config.enabled("psf"):
        if psf_strategy == "safe":
            pipeline1.add_step(
                SafePointSpreadFunction.extract,
                config={"safe_psf_config": safe_psf_config},
                step_name="Extract PSF",
                step_id="psf_extract",
                product_kind="psf",
            )
        else:
            pipeline1.add_step(
                PointSpreadFunction.extract,
                config={"psfconv_config": psfconv_config},
                step_name="Extract PSF",
                step_id="psf_extract",
                product_kind="psf",
            )
    if pipeline_config.enabled("psfconv") and psf_strategy == "safe":
        pipeline1.add_image_set_step(
            SafePointSpreadFunction.prepare,
            config={"safe_psf_config": safe_psf_config},
            step_name="Prepare Safe PSF Kernels",
            step_id="psf_prepare",
            product_kind="psf_match",
        )
    if pipeline_config.enabled("psfconv") and selected_pretrim_fov is not None:
        if psf_strategy == "safe":
            pipeline1.add_step(
                SafePointSpreadFunction.pretrim,
                config={"safe_psf_config": safe_psf_config},
                step_name="Pre-convolution Trim",
                step_id="psf_pretrim",
                mutates=("data", "header", "error"),
            )
        else:
            pipeline1.add_step(
                Register.pretrim_for_convolution,
                config={
                    "pretrim_fov": psfconv_config.pretrim_fov,
                    "kernel_truncate": psfconv_config.kernel_truncate,
                },
                step_name="Pre-convolution Trim",
                step_id="psf_pretrim",
                mutates=("data", "header", "error"),
            )
    gaia_mask_enabled and pipeline1.add_step(
        Masking.prepare_gaia_reference,
        config={"gaia_mask": gaia_mask},
        step_name="Prepare GAIA Mask Reference",
        step_id="gaia_reference",
        product_kind="gaia_reference",
    )
    if pipeline_config.enabled("psfconv"):
        if psf_strategy == "safe":
            convolution_engine = PSFConvolutionEngine(safe_psf_config.convolution_config())
            pipeline1.add_step(
                SafePointSpreadFunction.convolution,
                config={
                    "convolution_engine": convolution_engine,
                    "safe_psf_config": safe_psf_config,
                },
                step_name="Convolve with PSF",
                step_id="psf_convolve",
                mutates=("data", "header", "error"),
            )
        else:
            convolution_engine = PSFConvolutionEngine(psfconv_config)
            pipeline1.add_step(
                PointSpreadFunction.convolution,
                config={
                    "convolution_engine": convolution_engine,
                    "psfconv_config": psfconv_config,
                },
                step_name="Convolve with PSF",
                step_id="psf_convolve",
                mutates=("data", "error"),
            )

    
    pipeline_config.enabled("register") and pipeline1.add_image_set_step(
        Register.register_image_set,
        step_name="SWarp Registration",
        step_id="register",
        mutates=("data", "header", "error"),
    )
    pipeline_config.enabled("trim") and pipeline1.add_step(
        Register().trim,
        config={"trim_size": trim_size},
        step_name="Trimming",
        step_id="trim",
        mutates=("data", "header", "error"),
    )
    pipeline_config.enabled("mask") and pipeline1.add_step(
        Masking.adapt_mask,
        config={"manual": manual_mask, "gaia_mask": gaia_mask},
        step_name="Masking",
        step_id="mask",
        mutates=("data", "error"),
    )
    pipeline_config.enabled("skyinter") and pipeline1.add_step(
        interpolate_sky,
        step_name="Interpolate Masked Region",
        step_id="sky_interpolate",
        mutates=("data",),
    )
    pipeline_config.enabled("unit") and pipeline1.add_step(
        conversion().unitConvertor,
        step_name="Convert Unit",
        step_id="unit_convert",
        mutates=("data", "error", "header"),
    )
    
    pipeline_config.enabled("dered") and pipeline1.add_step(
        Reddening().dered,
        step_name="Dereddening",
        step_id="deredden",
        mutates=("data", "error"),
    )
    pipeline_config.enabled("bin") and pipeline1.add_step(
        Bin.do_binning,
        config={"bin_size": int(bin)},
        step_name="Binning Image",
        step_id="bin",
        mutates=("data", "error", "header"),
    )
    
    pipeline_config.enabled("cutout") and pipeline1.add_step(
        CutRegion.get_shape,
        config={"box_size": box_size, "cut_coeff": cut_coeff},
        step_name="Get Cutout Region",
        step_id="cutout_measure",
        product_kind="cutout_shape",
    )

    if (
        pipeline_config.enabled("background")
        and pipeline_config.enabled("cutout")
        and background_config.post_residual
        and not background_config.legacy_only
    ):
        pipeline1.add_image_set_step(
            residualBackgroundSubtraction,
            config={"background_config": background_config},
            step_name="Residual Background",
            step_id="residual_background",
            mutates=("data", "header"),
            product_kind="residual_background",
        )
    
    pipeline_config.enabled("cutout") and pipeline1.add_step(
        CutRegion.cutout_region,
        step_name="Cutout Image",
        step_id="cutout_apply",
        mutates=("data", "error"),
    )

    if checkpoint_manager is not None:
        if output_config.checkpoint_dir is not None:
            raise ValueError("checkpoint_dir cannot be changed while resuming a run")
        checkpoint_manager.validate_step_plan(pipeline1.pipeline_steps)
    elif output_config.save_checkpoints:
        science_settings = {
            "cat_type": cat_type,
            "pipeline_config": asdict(pipeline_config),
            "trim_size": trim_size,
            "manual_mask": manual_mask,
            "gaia_mask": gaia_mask,
            "bin": int(bin),
            "box_size": box_size,
            "cut_coeff": cut_coeff,
            "galaxy_metadata": galaxy_metadata,
            "metadata_config": (
                asdict(metadata_config) if metadata_config is not None else None
            ),
            "filter_config": filter_config,
            "psf_strategy": psf_strategy,
            "psfconv_config": (
                asdict(psfconv_config) if psfconv_config is not None else None
            ),
            "safe_psf_config": (
                asdict(safe_psf_config) if safe_psf_config is not None else None
            ),
            "background_config": asdict(background_config),
        }
        checkpoint_manager = CheckpointManager.create(
            galaxy_image_set,
            settings=science_settings,
            step_plan=CheckpointManager.step_plan(pipeline1.pipeline_steps),
            checkpoint_dir=output_config.checkpoint_dir,
        )
    
    galaxy_image_set = pipeline1.execute(
        plot_step=plot_step,
        verbose=verbose,
        output_config=output_config,
        checkpoint_manager=checkpoint_manager,
        completed_step_ids=completed_step_ids,
    )
    
    input_df = inputGenerator.dataframe_generator(
        galaxy_image_set,
        cat_type,
        metadata_resolver=metadata_resolver,
    )
    set_last_catalog = getattr(galaxy_image_set, "_set_last_catalog", None)
    if callable(set_last_catalog) and input_df is not None:
        flux_unit = "mJy" if pipeline_config.enabled("unit") else "native"
        set_last_catalog(CatalogFrame(
            data=input_df,
            catalog_type=cat_type,
            units={"flux": flux_unit, "error": flux_unit},
        ))
    if input_df is not None and checkpoint_manager is not None:
        input_df.attrs["spec7dt_run_id"] = next(
            iter(checkpoint_manager.manifests.values())
        )["run_id"]
        input_df.attrs["spec7dt_checkpoint_dirs"] = {
            galaxy: str(path)
            for galaxy, path in checkpoint_manager.output_dirs.items()
        }
    return input_df
