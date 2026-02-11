import pandas as pd
import numpy as np

from .utility import useful_functions
from ..handlers.filter_handler import Filters

class inputGenerator:
    def __init__(self):
        pass
    
    @classmethod
    def dataframe_generator(cls, image_set, cat_type):
        data_dict = {}
        generated_ids = False
        
        for i, ((galaxy, obs, band), values) in enumerate(useful_functions.tour_nested_dict_with_keys(image_set.data)):
            # With the first data, make structure of dataframe
            flat_values = np.asarray(values).flatten()
            
            if not generated_ids:
                # Generate IDs only once
                data_dict['id'] = [f"{galaxy}_{k}" for k in range(flat_values.size)]
                generated_ids = True
            
            data_dict[f"{obs}.{band}"] = flat_values
            data_dict[f"{obs}.{band}_err"] = np.asarray(image_set.error[galaxy][obs][band]).flatten()
        
        df = pd.DataFrame(data_dict)
        del data_dict  # Free memory immediately
        
        # Convert big-endian columns to native byte order
        for col in df.columns:
            if df[col].dtype.byteorder == '>':
                df[col] = df[col].astype(df[col].dtype.newbyteorder().type)
        
        float_cols = df.select_dtypes(include=['floating']).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype('float32')
            
        flux_cols = [col for col in float_cols if "_err" not in col]
            
        if flux_cols:
            flux_cut = (df[flux_cols] > 0).sum(axis=1) >= (len(flux_cols) / 2)
            df = df[flux_cut]

            err_cols = [col + "_err" for col in flux_cols]

            for flux_col, err_col in zip(flux_cols, err_cols):
                if err_col in df.columns:
                    mask = df[err_col] > 0.5 * df[flux_col]
                    df.loc[mask, [flux_col, err_col]] = np.nan
        
        df = df.astype({'id': 'str'})
        
        colnames = Filters.get_catcols(cat_type, float_cols)
        df.rename(columns=colnames, inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        
        df['redshift'] = np.nan
        galaxies = list(image_set.data.keys())
        for g in galaxies:
            z = useful_functions.get_redshift(g)
            if z is not None:
                # Use startswith for faster matching than contains
                mask = df['id'].str.startswith(f"{g}_")
                df.loc[mask, 'redshift'] = z
            
        return df