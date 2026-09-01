import inspect
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass, field

from ..plot.plot import DrawGalaxy
from ..handlers.filter_handler import Filters
from ..core import CatalogFrame
from .metadata import GalaxyMetadataConfig, GalaxyMetadataResolver
from .reporting import RichPipelineReporter, reporter_context


@dataclass(frozen=True)
class PipelineOutputConfig:
    progress: bool = True
    show_alerts: bool = True
    pdf_path: str | Path | None = None
    pdf_dpi: int = 300
    show_plots: bool | None = None
    overwrite: bool = False

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
    
    def add_step(self, function, config=None, step_name=None):
        """파이프라인에 처리 단계 추가"""
        if step_name is None:
            step_name = function.__name__
        
        self.pipeline_steps.append({
            'name': step_name,
            'function': function,
            'config': config or {},
            'scope': 'image',
        })

    def add_image_set_step(self, function, config=None, step_name=None):
        """Add a step that runs once against the complete GalaxyImageSet."""
        if step_name is None:
            step_name = function.__name__

        self.pipeline_steps.append({
            'name': step_name,
            'function': function,
            'config': config or {},
            'scope': 'image_set',
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
    ):
        """파이프라인 실행"""
        output_config = PipelineOutputConfig.from_value(output_config)
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
        )
        timings = []
        active_step = None
        active_target = None
        succeeded = False
        reporter.start(len(self.pipeline_steps))

        with reporter_context(reporter):
            try:
                for step in self.pipeline_steps:
                    active_step = step["name"]
                    active_target = None
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
                        delegated_progress = self._execute_image_set_step(
                            step,
                            verbose,
                            reporter=reporter,
                        )
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
                            elapsed = self._execute_step(
                                step,
                                galaxy,
                                observatory,
                                band,
                                image,
                                header,
                                error,
                                verbose,
                            )
                            reporter.detail(
                                f"✓ {step['name']} completed for {active_target} "
                                f"in {elapsed:.3f}s"
                            )
                            reporter.advance_target()
                            processed_targets += 1

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
            reporter.summary(timings, pdf_path=pdf_path)
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
        function(**filtered_kwargs)
        elapsed = perf_counter() - target_start
        return elapsed

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
        function(**filtered_kwargs)
        elapsed = perf_counter() - target_start
        if reporter is not None:
            reporter.detail(
                f"✓ {step['name']} completed for GalaxyImageSet in {elapsed:.3f}s"
            )
        return "reporter" in sig.parameters


from ..reduction.registration import Register
from .unit import conversion
from ..reduction.background import (
    BackgroundConfig,
    backgroundSubtraction,
    estimateMissingError,
    has_missing_error_maps,
)
from ..reduction.PSF import (
    PSFConvolutionConfig,
    PSFConvolutionEngine,
    PointSpreadFunction,
)
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
    galaxy_image_set,
    cat_type,
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
    background_config: BackgroundConfig | dict = None,
    output_config: PipelineOutputConfig | dict = None,
):
    pipeline_config = PipelineConfig.from_processes(config if config is not None else processes)
    psfconv_config = PSFConvolutionConfig.from_value(psfconv_config)
    background_config = BackgroundConfig.from_value(background_config)
    output_config = PipelineOutputConfig.from_value(output_config)
    if output_config.pdf_path is not None and not isinstance(plot_step, dict):
        raise ValueError("pdf_path requires plot_step to be a selector dictionary")
    if not pipeline_config.enabled("psfconv") and psfconv_config.pretrim_fov is not None:
        raise ValueError("pretrim_fov requires the psfconv pipeline step to be enabled")
    manual_mask = Masking.prepare_manual_mask(manual_mask)
    if isinstance(metadata_config, dict):
        metadata_config = GalaxyMetadataConfig(**metadata_config)
    metadata_resolver = GalaxyMetadataResolver(metadata=galaxy_metadata, config=metadata_config)

    filter_options = {"allow_svo": True, "cache": True, "warn": True, "unknown_policy": "best_effort"}
    if filter_config:
        filter_options.update(filter_config)

    gaia_mask_enabled = pipeline_config.enabled("mask") and gaia_mask is not None and gaia_mask is not False
    if pipeline_config.enabled("dered") or pipeline_config.enabled("psf") or gaia_mask_enabled:
        Filters.ensure_filters_for_image_set(galaxy_image_set, **filter_options)
    if gaia_mask_enabled:
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
        )
    elif has_missing_error_maps(galaxy_image_set):
        pipeline1.add_step(
            estimateMissingError,
            config={"background_config": background_config},
            step_name="Estimate Missing Error",
        )
    pipeline_config.enabled("psf") and pipeline1.add_step(
        PointSpreadFunction.extract,
        config={"psfconv_config": psfconv_config},
        step_name="Extract PSF",
    )
    if pipeline_config.enabled("psfconv") and psfconv_config.pretrim_fov is not None:
        pipeline1.add_step(
            Register.pretrim_for_convolution,
            config={
                "pretrim_fov": psfconv_config.pretrim_fov,
                "kernel_truncate": psfconv_config.kernel_truncate,
            },
            step_name="Pre-convolution Trim",
        )
    gaia_mask_enabled and pipeline1.add_step(
        Masking.prepare_gaia_reference,
        config={"gaia_mask": gaia_mask},
        step_name="Prepare GAIA Mask Reference",
    )
    if pipeline_config.enabled("psfconv"):
        convolution_engine = PSFConvolutionEngine(psfconv_config)
        pipeline1.add_step(
            PointSpreadFunction.convolution,
            config={
                "convolution_engine": convolution_engine,
                "psfconv_config": psfconv_config,
            },
            step_name="Convolve with PSF",
        )

    
    pipeline_config.enabled("register") and pipeline1.add_image_set_step(
        Register.register_image_set,
        step_name="SWarp Registration",
    )
    pipeline_config.enabled("trim") and pipeline1.add_step(Register().trim, config={"trim_size" : trim_size}, step_name="Trimming")
    pipeline_config.enabled("mask") and pipeline1.add_step(
        Masking.adapt_mask,
        config={"manual": manual_mask, "gaia_mask": gaia_mask},
        step_name="Masking",
    )
    pipeline_config.enabled("skyinter") and pipeline1.add_step(interpolate_sky, step_name="Interpolate Masked Region")
    pipeline_config.enabled("unit") and pipeline1.add_step(conversion().unitConvertor, step_name="Convert Unit")
    
    pipeline_config.enabled("dered") and pipeline1.add_step(Reddening().dered, step_name="Dereddening")
    pipeline_config.enabled("bin") and pipeline1.add_step(Bin.do_binning, config={"bin_size": int(bin)}, step_name="Binning Image")
    
    pipeline_config.enabled("cutout") and pipeline1.add_step(CutRegion.get_shape, config={"box_size" : box_size, "cut_coeff": cut_coeff}, step_name="Get Cutout Region")
    
    pipeline_config.enabled("cutout") and pipeline1.add_step(CutRegion.cutout_region, step_name="Cutout Image")
    
    galaxy_image_set = pipeline1.execute(
        plot_step=plot_step,
        verbose=verbose,
        output_config=output_config,
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
    return input_df
