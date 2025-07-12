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
        input_df = pd.DataFrame(columns=['id'])
        
        for i, ((galaxy, obs, band), values) in enumerate(useful_functions.tour_nested_dict_with_keys(image_set.data)):
            # With the first data, make structure of dataframe
            if i == 0:
                x, y = values.shape
                ids = pd.DataFrame([f"{galaxy}_{i}" for i in range(int(x * y))], columns=["id"])
                input_df = pd.concat([input_df, ids])
            else:
                pass
            
            im_series = pd.Series(np.array(values).flatten(), name=f"{obs}.{band}")
            input_df = pd.concat([input_df, im_series], axis=1)
            
            err_series = pd.Series(np.array(image_set.error[galaxy][obs][band]).flatten(), name=f"{obs}.{band}_err")
            input_df = pd.concat([input_df, err_series], axis=1)
            
        # drop NaN values
        float_cols = input_df.select_dtypes(include=['floating']).columns
        input_df[float_cols] = input_df[float_cols].astype('float64')  

            
        input_df.loc[:, input_df.columns != 'id'] = input_df.loc[:, input_df.columns != 'id'].where(input_df.loc[:, input_df.columns != 'id'] >= 0, 0).fillna(0)
        input_df = input_df.astype({'id': 'str'})
        input_df = input_df[(input_df.loc[:, input_df.columns != 'id'] != 0).all(axis=1)]
        
        colnames = Filters.get_catcols(cat_type)
        input_df.rename(columns=colnames, inplace=True)
            
        return input_df

"""input_df = pd.DataFrame(columns=['id']+BANDs)

for id in list(set([i.split('-')[0] for i in region_dic.keys()])):
    ids = pd.DataFrame([f'{id}_{i}' for i in range(bin_size ** 2)], columns=['id'])
    input_df = pd.concat([input_df, ids])
    
for i, name in enumerate(list(region_dic.keys())):
    id = name.split('-')[0]
    band = name.split('-')[2]
    flatten_img = np.array(region_dic[name]).flatten()
    
    input_df.loc[input_df['id'].str.contains(id), band] = flatten_img
    
    if err_bool[name]:
        if f'{band}_err' not in input_df.columns:
            input_df[f'{band}_err'] = ''
        flatten_img = np.array(region_err_dic[name]).flatten()
        input_df.loc[input_df['id'].str.contains(id), f'{band}_err'] = flatten_img
    else:
        input_df.loc[input_df['id'].str.contains(id), f'{band}_err'] = flatten_img * 0.1

input_df = input_df.astype('float64', errors='ignore')
# input_df = input_df.drop(['F657N', 'F658N'], axis=1)

# nan_counts = input_df.isnull().sum(axis=1)
# neg_counts = input_df[input_df.loc[:, (input_df.columns != 'id')] <= 0].sum(axis=1)
# input_df[(nan_counts > 1) | (neg_counts < 0), input_df.columns != 'id'] = 0
input_df.loc[:, input_df.columns != 'id'] = input_df.loc[:, input_df.columns != 'id'].where(input_df.loc[:, input_df.columns != 'id'] >= 0, 0).fillna(0)
input_df = input_df.astype({'id': 'str'})

for _, key in enumerate(redshifting.keys()):
    input_df.loc[input_df['id'].str.contains(key), 'redshift'] = redshifting[key]

print(input_df)
# input_df = input_df.drop(columns = ['m550', 'm575'])
input_df = input_df[(input_df.loc[:, input_df.columns != 'id'] != 0).all(axis=1)]



band_for_galapy = {
    'u': 'SDSS.u',
    'g': 'SDSS.g',
    'r': 'SDSS.r',
    'i': 'SDSS.i',
    'z': 'SDSS.z',
    'ch1': 'Spitzer.IRAC.I1',
    'ch2': 'Spitzer.IRAC.I2',
    'F657N': 'HST.WFC3.UVIS1.F657N',
    'F658N': 'HST.WFC3.UVIS1.F658N',
    'u_err': 'SDSS.u_err',
    'g_err': 'SDSS.g_err',
    'r_err': 'SDSS.r_err',
    'i_err': 'SDSS.i_err',
    'z_err': 'SDSS.z_err',
    'ch1_err': 'Spitzer.IRAC.I1_err',
    'ch2_err': 'Spitzer.IRAC.I2_err',
    'F657N_err': 'HST.WFC3.UVIS1.F657N_err',
    'F658N_err': 'HST.WFC3.UVIS1.F658N_err'
}
band_for_7DT = {f'm{int(wav)}':f'7DT.m{int(wav)}' for wav in np.linspace(400, 875, 20)}
band_for_cigale.update(band_for_7DT)
band_for_cigale.update({f'{key}_err': f'{band_for_cigale[key]}_err' for key in band_for_cigale.keys() if '_err' not in key})

input_df.rename(columns=band_for_cigale, inplace=True)


# sel_id = [f'{obj}_{id * bin_size + 73 + k}' for obj in ['NGC3627', 'NGC4254', 'NGC4303', 'NGC4535'] for id in range(bin_size) for k in range(6)]
# input_df = input_df[input_df.id.isin(sel_id)]
input_df.reset_index()

with pd.option_context('display.max_rows', 500, 'display.max_columns', 20): 
    print(input_df)
tbl = Table.from_pandas(input_df)
date_now = time.strftime('%Y%m%d-%H%M', time.localtime())
tbl.write(f'6_CIGALE_inputs/{date_now}_cigale_table.fits', overwrite=True)

end_time = datetime.datetime.now()
print(f'Start at: {start_time}')
print(f'End at: {end_time}')
print(f'Time taken: {end_time - start_time}')"""