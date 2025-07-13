import time
import pandas as pd
import numpy as np

from astropy.table import Table

from .utility import Filters, useful_functions

class inputGenerator:
    def __init__(self):
        pass
    
    @classmethod
    def dataframe_generator(cls, image_set, cat_type):
        df = pd.DataFrame(columns=['id'])
        
        for i, ((galaxy, obs, band), values) in enumerate(useful_functions.tour_nested_dict_with_keys(image_set.data)):
            # With the first data, make structure of dataframe
            if i == 0:
                x, y = values.shape
                ids = pd.DataFrame([f"{galaxy}_{i}" for i in range(int(x * y))], columns=["id"])
                df = pd.concat([df, ids])
            else:
                pass
            
            im_series = pd.Series(np.array(values).flatten(), name=f"{obs}.{band}")
            df = pd.concat([df, im_series], axis=1)
            
            err_series = pd.Series(np.array(image_set.error[galaxy][obs][band]).flatten(), name=f"{obs}.{band}_err")
            df = pd.concat([df, err_series], axis=1)
        
        float_cols = df.select_dtypes(include=['floating']).columns
        df[float_cols] = df[float_cols].astype('float64')
        flux_cols = [col for col in float_cols if "_err" not in col]
            
        df_filtered = (df[flux_cols] > 0).all(axis=1)
        df = df[df_filtered]
        
        df = df.astype({'id': 'str'})
        
        colnames = Filters.get_catcols(cat_type, float_cols)
        df.rename(columns=colnames, inplace=True)
        
        df.reset_index()
            
        return df