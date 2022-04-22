import os
import fiona
import datetime
import numpy as np
import pandas as pd
import geopandas as gp
import rasterio as rio

from tqdm import tqdm
from rasterio.mask import mask


def read_and_mask(fn,area_geom,scale_factor):
	
	src = rio.open(fn) # Read file
	src2 = rio.mask.mask(src, area_geom, crop=True) # Clip to shp 

	fl_arr = src2[0].astype(np.float) # read as array
	arr = fl_arr.reshape(fl_arr.shape[1], fl_arr.shape[2]) / scale_factor # Reshape bc rasterio has a different dim ordering 
	outarr = np.where(arr < 0 ,np.nan, arr)# Mask nodata vals 
	arr = arr/scale_factor # divide by scale factor

	return outarr



def main():
	
	# Read GDF
	gdf = gp.read_file("../shape/sierra_catchments.shp")

	# Setup dirs 
	smlt_dir = "../data/SMLT/"
	prcp_dir = "../data/PLQD/"

	# Get lists of files 
	smlt_files = [os.path.join(smlt_dir,x) for x in os.listdir(smlt_dir) if x.endswith(".tif")]
	prcp_files = [os.path.join(prcp_dir,x) for x in os.listdir(prcp_dir) if x.endswith(".tif")]

	print(len(smlt_files))

	# Sort
	smlt_files.sort()
	prcp_files.sort()

	# Datetime the start/end
	start = datetime.datetime.strptime("2003-10-01", "%Y-%m-%d")
	end = datetime.datetime.strptime("2021-09-30", "%Y-%m-%d")
	dt_idx = pd.date_range(start,end, freq='D')

	# Datetime objs --> strs
	d1strs = [x.strftime('%Y%m%d') for x in dt_idx]

	# Read catchments 
	gdf = gp.read_file("../shape/sierra_catchments.shp")
	stn_lookup = dict(zip(gdf['stid'], [x[:3] for x in gdf['catch_name']]))

	# Loop through watersheds
	stn_id_list = [x for x in list(gdf['stid']) if "MCR" not in x if "CFW" not in x]

	for stn_id in stn_id_list:

		print("PROCESSING {}  = =====================".format(stn_id))

		# Read shapefile for mask
		shppath = "../shape/{}.shp".format(stn_id)

		with fiona.open(shppath, "r") as shapefile:
			area_geom = [feature["geometry"] for feature in shapefile]
		
		smlt_arrs = {}
		prcp_arrs = {}

		# Make a blank array for the dates that don't exist 
		ref_src = rio.open(smlt_files[0])
		ref_src_masked = mask(ref_src, area_geom, crop=True)[0] # Clip to shp 
		ref_arr_masked = ref_src_masked.reshape(ref_src_masked.shape[1], ref_src_masked.shape[2]) 
		ref_arr_0 = np.zeros_like(ref_arr_masked)
		ref_arr = np.where(ref_arr_0==0,np.nan,ref_arr_0)

		# Loop through each day 
		for datestr in tqdm(d1strs[:]):

			smlt_fn = [x for x in smlt_files if datestr in x]
			prcp_fn = [x for x in prcp_files if datestr in x]

			 # Add nan ims for the dates with no data (listed in SI)
			if len(smlt_fn) == 0:
				print("SMLT {} MISSING".format(datestr))
				smlt_arr = ref_arr.copy()
			else:
				smlt_arr = read_and_mask(smlt_fn[0],area_geom,scale_factor= 100000)
			
			if len(prcp_fn) == 0:
				print("PRCP {} MISSING".format(datestr))
				prcp_arr = ref_arr.copy()
			else:
				prcp_arr = read_and_mask(prcp_fn[0],area_geom, scale_factor= 10)

			smlt_arrs[datestr] = smlt_arr
			prcp_arrs[datestr] = prcp_arr

		smlt_out = np.dstack(list(smlt_arrs.values()))
		prcp_out = np.dstack(list(prcp_arrs.values()))

		np.save("../data/Watersheds/{}_smlt.npy".format(stn_id), smlt_out )
		np.save("../data/Watersheds/{}_prcp.npy".format(stn_id), prcp_out )

		print("********" *5)
		print("WROTE FILES FOR {} in ../data/Watersheds ".format(stn_id))
		print("********" *5)


if __name__ == '__main__':
	main()
			
