import inspect
import numpy as np

class ImageProcessingPipeline:
    def __init__(self, galaxy_image_set):
        self.galaxy_image_set = galaxy_image_set
        self.pipeline_steps = []
        self.step_configs = {}
    
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
    
    def execute(self, galaxy_filter=None, observatory_filter=None, band_filter=None):
        """파이프라인 실행"""
        # 필터 조건에 맞는 이미지들을 선택
        targets = self._get_filtered_targets(galaxy_filter, observatory_filter, band_filter)
        
        for galaxy, observatory, band, image, header, error in targets:
            for step in self.pipeline_steps:
                self._execute_step(step, galaxy, observatory, band, image, header, error)
    
    def _get_filtered_targets(self, galaxy_filter, observatory_filter, band_filter):
        """필터 조건에 맞는 (galaxy, observatory, band) 조합 반환"""
        targets = []
        
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
                    
                    targets.append((galaxy, observatory, band,
                                    target_image, target_header, target_error))
        
        return targets
    
    def _execute_step(self, step, galaxy, observatory, band, image, header, error):
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
        print(f"✓ {step['name']} completed for {galaxy}/{observatory}/{band}")
        
        # try:
        #     result = function(**filtered_kwargs)
        #     print(f"✓ {step['name']} completed for {galaxy}/{observatory}/{band}")
        # except Exception as e:
        #     print(f"✗ {step['name']} failed for {galaxy}/{observatory}/{band}: {e}")


from .unit import conversion
from ..reduction.background import backgroundSubtraction
from ..reduction.PSF import PointSpreadFunction
from ..manipulation.reddening import Reddening
from ..manipulation.mask import Masking
from ..manipulation.sky_interpolate import interpolate_sky
from ..division.binning import Bin
from ..division.cutout import CutRegion

from .file_generator import inputGenerator

def execute_pipeline(galaxy_image_set, bin, cat_type):
    pipeline1 = ImageProcessingPipeline(galaxy_image_set)

    pipeline1.add_step(conversion().unitConvertor, step_name="Convert Unit")
    # pipeline1.add_step(backgroundSubtraction)
    pipeline1.add_step(PointSpreadFunction.extract, step_name="Extract PSF")
    
    pipeline1.execute()
    
    pipeline2 = ImageProcessingPipeline(galaxy_image_set)
    
    pipeline2.add_step(PointSpreadFunction.convolution, step_name="Convolve with PSF")
    pipeline2.add_step(Reddening().dered, step_name="Dereddening")
    pipeline2.add_step(Masking.adapt_mask, step_name="Masking")
    pipeline2.add_step(interpolate_sky, step_name="Interpolate Masked Region")
    pipeline2.add_step(Bin.do_binning, config={"bin_size": int(bin)}, step_name="Binning Image")
    
    pipeline2.execute()
    
    pipeline3 = ImageProcessingPipeline(galaxy_image_set)
    
    pipeline3.add_step(CutRegion.cutout_region, config={"box_size" : None}, step_name="Cutout Image Region")
    
    pipeline3.execute()
    
    input_df = inputGenerator.dataframe_generator(galaxy_image_set, cat_type)
    return input_df
