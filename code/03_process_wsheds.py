import os
import fiona
import datetime
import numpy as np
import pandas as pd
import geopandas as gp
import rasterio as rio

from numba import jit 
from tqdm import tqdm
from rasterio.mask import mask

from numba import jit

@jit(nopython=True)
def interp3d(arr_3d):
    result=np.zeros_like(arr_3d)
    for i in range(arr_3d.shape[0]):
        for j in range(arr_3d.shape[1]):
            arr=arr_3d[i,j,:]
            # If all elements are nan then cannot conduct linear interpolation.
            if np.sum(np.isnan(arr))==arr.shape[0]:
                result[i,j,:]=arr
            else:
                # If the first elemet is nan, then assign the value of its right nearest neighbor to it.
                if np.isnan(arr[0]):
                    arr[0]=arr[~np.isnan(arr)][0]
                # If the last element is nan, then assign the value of its left nearest neighbor to it.
                if np.isnan(arr[-1]):
                    arr[-1]=arr[~np.isnan(arr)][-1]
                # If the element is in the middle and its value is nan, do linear interpolation using neighbor values.
                for k in range(arr.shape[0]):
                    if np.isnan(arr[k]):
                        x=k
                        x1=x-1
                        x2=x+1
                        # Find left neighbor whose value is not nan.
                        while x1>=0:
                            if np.isnan(arr[x1]):
                                x1=x1-1
                            else:
                                y1=arr[x1]
                                break
                        # Find right neighbor whose value is not nan.
                        while x2<arr.shape[0]:
                            if np.isnan(arr[x2]):
                                x2=x2+1
                            else:
                                y2=arr[x2]
                                break
                        # Calculate the slope and intercept determined by the left and right neighbors.
                        slope=(y2-y1)/(x2-x1)
                        intercept=y1-slope*x1
                        # Linear interpolation and assignment.
                        y=slope*x+intercept
                        arr[x]=y
                result[i,j,:]=arr
    return result

def main(stn_id):

	print("=======" * 15)
	print("PROCESSING: {}".format(stn_id))
	print("=======" * 15)

	# Set start / end date
	dt_idx = pd.date_range('2003-10-01','2021-09-30', freq='D')

	# Set filepaths for hydro data 
	smlt_fn = os.path.join('../data/Watersheds/{}_smlt.npy'.format(stn_id))
	prcp_fn = os.path.join('../data/Watersheds/{}_prcp.npy'.format(stn_id))
	
	outdirs = ["../data/Watersheds/1d","../data/Watersheds/3d","../data/Watersheds/5d"]
	for ext_dir in outdirs:
		if not os.path.exists(ext_dir):
			os.mkdir(ext_dir)

	smlt_fn_1d = "../data/Watersheds/1d/{}_1d_smlt".format(stn_id)
	prcp_fn_1d = "../data/Watersheds/1d/{}_1d_prcp".format(stn_id)

	smlt_fn_3d = "../data/Watersheds/3d/{}_3d_smlt".format(stn_id)
	prcp_fn_3d = "../data/Watersheds/3d/{}_3d_prcp".format(stn_id)
	
	smlt_fn_5d = "../data/Watersheds/5d/{}_5d_smlt".format(stn_id)
	prcp_fn_5d = "../data/Watersheds/5d/{}_5d_prcp".format(stn_id)
	

	print("Interpolating missing dates in time dimension   =================")

	if not os.path.exists(smlt_fn_1d):
		smlt_1d = interp3d(np.load(smlt_fn))
		np.save(smlt_fn_1d, smlt_1d)
		print("WROTE ============= {}".format(smlt_fn_1d))
	else:
		smlt_1d = np.load(smlt_fn_1d)

	if not os.path.exists(prcp_fn_1d):
		prcp_1d = interp3d(np.load(prcp_fn))
		np.save(prcp_fn_1d, prcp_1d)
		print("WROTE ============= {}".format(prcp_fn_1d))
	else:
		prcp_1d = np.load(prcp_fn_1d)

	print('convolving 3D arrays =================')
	
	if not os.path.exists(smlt_fn_3d):
		conv_win_3d = np.ones(3)
		smlt_rolling_3d = np.apply_along_axis(lambda m: np.convolve(m, conv_win_3d, mode='full'), axis=2, arr=smlt_1d)
		smlt_3d = smlt_rolling_3d[:,:,2:]
		np.save(smlt_fn_3d, smlt_3d)
		print("WROTE ============= {}".format(smlt_fn_3d))
	else:
		print("{} ALREADY EXISTS, SKIPPING ============= {}".format(smlt_fn_3d))
		# smlt_3d = np.load(smlt_fn_3d)

	if not os.path.exists(prcp_fn_3d):
		conv_win_3d = np.ones(3)
		prcp_rolling_3d = np.apply_along_axis(lambda m: np.convolve(m, conv_win_3d, mode='full'), axis=2, arr=prcp_1d)
		prcp_3d = prcp_rolling_3d[:,:,2:]
		np.save(prcp_fn_3d, prcp_3d)
		print("WROTE ============= {}".format(prcp_fn_3d))
	else:
		print("{} ALREADY EXISTS, SKIPPING ============= {}".format(prcp_fn_3d))
		# prcp_3d = np.load(prcp_fn_3d)
	
	print('convolving 5d arrays =================')
	
	if not os.path.exists(smlt_fn_5d):
		conv_win_5d = np.ones(5)
		smlt_rolling_5d = np.apply_along_axis(lambda m: np.convolve(m, conv_win_5d, mode='full'), axis=2, arr=smlt_1d)
		smlt_5d = smlt_rolling_5d[:,:,4:]
		np.save(smlt_fn_5d, smlt_5d)
		print("WROTE ============= {}".format(smlt_fn_5d))
	else:
		print("{} ALREADY EXISTS, SKIPPING ============= {}".format(smlt_fn_5d))
		# smlt_5d = np.load(smlt_fn_5d)

	if not os.path.exists(prcp_fn_5d):
		conv_win_5d = np.ones(5)
		prcp_rolling_5d = np.apply_along_axis(lambda m: np.convolve(m, conv_win_5d, mode='full'), axis=2, arr=prcp_1d)
		prcp_5d = prcp_rolling_5d[:,:,4:]
		np.save(prcp_fn_5d, prcp_5d)
		print("WROTE ============= {}".format(prcp_fn_5d))
	else:
		print("{} ALREADY EXISTS, SKIPPING ============= {}".format(prcp_fn_5d))
		# prcp_5d = np.load(prcp_fn_5d)

	print("PRCP OUTFILES SHAPES")
	print(len(dt_idx), prcp_1d.shape[2], prcp_3d.shape[2], prcp_5d.shape[2])
	print("SMLT OUTFILES SHAPES")
	print(len(dt_idx), smlt_1d.shape[2], smlt_3d.shape[2], smlt_5d.shape[2])


if __name__ == '__main__':

	gdf = gp.read_file("../shape/sierra_catchments.shp")

	stids_all = list(gdf['stid'].values)
	nodata_stids = ["MCR", "CFW", "NHG"]

	stids = [x for x in stids_all if x not in nodata_stids]

	for stid in tqdm(stids):
		main(stid)
