import inspect
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from datetime import datetime
import pickle

from ..plot.plot import DrawGalaxy

class ImageProcessingPipeline:
    def __init__(self, galaxy_image_set):
        self.galaxy_image_set = galaxy_image_set
        self.pipeline_steps = []
        self.step_configs = {}
        self.starttime = datetime.now()
    
    def add_step(self, function, config=None, step_name=None):
        """파이프라인에 처리 단계 추가"""
        if step_name is None:
            step_name = function.__name__
        
        self.pipeline_steps.append({
            'name': step_name,
            'function': function,
            'config': config or {}
        })
        
    def remove_step(self, step_name):
        """특정 단계 제거"""
        self.pipeline_steps = [step for step in self.pipeline_steps 
                              if step['name'] != step_name]
    
    def execute(self, galaxy_filter=None, observatory_filter=None, band_filter=None, plot_step=False, verbose=False):
        """파이프라인 실행"""
        # 필터 조건에 맞는 이미지들을 선택
        
        for step in self.pipeline_steps:
            # Use generator to process images one by one without holding all in memory
            for galaxy, observatory, band, image, header, error in self._get_filtered_targets(galaxy_filter, observatory_filter, band_filter):
                self._execute_step(step, galaxy, observatory, band, image, header, error, verbose)
                
            if plot_step is not None:
                print(f"Time lack: {datetime.now() - self.starttime}")
                self.starttime = datetime.now()
                
                # with open(f"Galaxy_Set_{step['name']}.pkl", 'wb') as file:
                #     pickle.dump(self.galaxy_image_set, file)
                
                print(galaxy, observatory, band)
                
                if isinstance(plot_step, dict):
                    if "band" in plot_step.keys():
                        DrawGalaxy.single_galaxy(self.galaxy_image_set, 
                                                plot_step.get("galaxy", galaxy),
                                                plot_step.get("obs", observatory),
                                                plot_step.get("band", band))
                    else:
                        DrawGalaxy.plot_galaxies(self.galaxy_image_set, galaxy, step["name"])
                        plt.show()
                
        return self.galaxy_image_set
    
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
                    except:
                        target_error = np.zeros_like(target_image)
                    
                    yield (galaxy, observatory, band, target_image, target_header, target_error)
    
    def _execute_step(self, step, galaxy, observatory, band, image, header, error, verbose):
        """Execute each step"""
        function = step['function']
        config = step['config']
        
        sig = inspect.signature(function)
        
        # image_data, header, error_data, galaxy_name, observatory, band, image_set
        kwargs = {'image_set': self.galaxy_image_set,
                  'image_data':image, 'header':header, 'error_data':error,
                  'galaxy_name': galaxy, 'observatory': observatory, 'band': band}
        
        kwargs.update(config)
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        result = function(**filtered_kwargs)
        print(f"✓ {step['name']} completed for {galaxy}/{observatory}/{band} {datetime.now() - self.starttime}")
        
        # try:
        #     result = function(**filtered_kwargs)
        #     print(f"✓ {step['name']} completed for {galaxy}/{observatory}/{band}") if verbose else None
        # except Exception as e:
        #     print(f"✗ {step['name']} failed for {galaxy}/{observatory}/{band}: {e}")
        
        return self.galaxy_image_set


from ..reduction.registration import Register
from .unit import conversion
from ..reduction.background import backgroundSubtraction
from ..reduction.PSF import PointSpreadFunction
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
    manual_mask: dict = None,
    bin=1,
    box_size=None,
    cut_coeff=1.5,
    config: PipelineConfig = None,
):
    pipeline_config = PipelineConfig.from_processes(config if config is not None else processes)
    
    
    pipeline1 = ImageProcessingPipeline(galaxy_image_set)

    pipeline_config.enabled("background") and pipeline1.add_step(backgroundSubtraction)
    pipeline_config.enabled("psf") and pipeline1.add_step(PointSpreadFunction.extract, step_name="Extract PSF")
    pipeline_config.enabled("psfconv") and pipeline1.add_step(PointSpreadFunction.convolution, step_name="Convolve with PSF")
    
    pipeline_config.enabled("register") and pipeline1.add_step(Register().save_all_progress, step_name="Save Progress")
    pipeline_config.enabled("register") and pipeline1.add_step(Register().swarp_register, step_name="SWarp Reigistration")
    pipeline_config.enabled("trim") and pipeline1.add_step(Register().trim, config={"trim_size" : trim_size}, step_name="Trimming")
    pipeline_config.enabled("unit") and pipeline1.add_step(conversion().unitConvertor, step_name="Convert Unit")
    
    pipeline_config.enabled("dered") and pipeline1.add_step(Reddening().dered, step_name="Dereddening")
    pipeline_config.enabled("mask") and pipeline1.add_step(Masking.adapt_mask, config={"manual": manual_mask}, step_name="Masking")
    pipeline_config.enabled("skyinter") and pipeline1.add_step(interpolate_sky, step_name="Interpolate Masked Region")
    pipeline_config.enabled("bin") and pipeline1.add_step(Bin.do_binning, config={"bin_size": int(bin)}, step_name="Binning Image")
    
    pipeline_config.enabled("cutout") and pipeline1.add_step(CutRegion.get_shape, config={"box_size" : box_size, "cut_coeff": cut_coeff}, step_name="Get Cutout Region")
    
    pipeline_config.enabled("cutout") and pipeline1.add_step(CutRegion.cutout_region, step_name="Cutout Image")
    
    galaxy_image_set = pipeline1.execute(plot_step=plot_step, verbose=verbose)
    
    input_df = inputGenerator.dataframe_generator(galaxy_image_set, cat_type)
    return input_df
