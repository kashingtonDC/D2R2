import os
import datetime

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gp
import multiprocessing as mp

from osgeo import gdal, osr
from tqdm import tqdm
from sklearn import metrics
from dateutil.relativedelta import relativedelta
from scipy import stats, spatial, signal, fftpack

import warnings
warnings.filterwarnings("ignore")

# Functions

def write_raster(array,gdf,outfn):
	'''
	converts a numpy array and a geopandas gdf to a geotiff
	Data values are stored in np.array
	spatial coordinates stored in gdf
	outfn - outpath
	'''

	xmin, ymin = gdf.bounds.minx.values[0], gdf.bounds.miny.values[0]
	xmax, ymax = gdf.bounds.maxx.values[0], gdf.bounds.maxy.values[0]
	nrows, ncols = array.shape
	xres = (xmax-xmin)/float(ncols)
	yres = (ymax-ymin)/float(nrows)
	geotransform =(xmin,xres,0,ymax,0, -yres)   

	output_raster = gdal.GetDriverByName('GTiff').Create(outfn,ncols, nrows, 1 , gdal.GDT_Float32)  # Open the file
	output_raster.SetGeoTransform(geotransform)  # Specify coords
	srs = osr.SpatialReference()                 # Establish encoding
	srs.ImportFromEPSG(4326)                     # WGS84 lat long
	output_raster.SetProjection(srs.ExportToWkt() )   # Export coordinate system 
	output_raster.GetRasterBand(1).WriteArray(array)   # Write array to raster

	print("wrote {}".format(outfn))
	return outfn


def calc_nbins(N):

	'''
	A. Hacine-Gharbi, P. Ravier, "Low bias histogram-based estimation of mutual information for feature selection", Pattern Recognit. Lett (2012).
	'''
	ee = np.cbrt(8 + 324*N + 12*np.sqrt(36*N + 729*N**2))
	bins = np.round(ee/6 + 2/(3*ee) + 1/3)

	return int(bins)

def calc_mi(imstack, inflow):

	# Build the out image
	mi_im = np.zeros_like(np.mean(imstack, axis = 2))

	rows, cols, time = imstack.shape
	px_ts = []
	rclist = []

	# extract pixelwise timeseries
	for row in range(rows):
		for col in range(cols):
			ts_arr = imstack[row,col,:]

			if not np.isnan(ts_arr).all():
				px_ts.append(pd.Series(ts_arr))
				rclist.append([row,col])
			else:
				px_ts.append(pd.Series(np.zeros_like(ts_arr)))
				rclist.append([row,col])

	pxdf = pd.concat(px_ts, axis = 1)
	pxdf.columns = pxdf.columns.map(str)

	# Populate the per-pixel lags 
	for rc, dfcolidx in tqdm(list(zip(rclist,pxdf.columns))):

		tempdf = pd.DataFrame([pxdf[dfcolidx].copy(),inflow]).T
		tempdf.columns = ['var','q']

		# get n bins
		nbins = calc_nbins(len(tempdf))

		# compute mutual info
		try: 
			mi = metrics.mutual_info_score(tempdf['var'].value_counts(normalize=True,bins = nbins),tempdf['q'].value_counts(normalize=True,bins = nbins))
		except:
			mi = np.nan

		# fill ims
		rowidx, colidx = rc
		mi_im[rowidx,colidx] = mi

	return mi_im

def normalize(x):
	return(x-np.nanmin(x))/(np.nanmax(x)- np.nanmin(x))

def calc_xcorr_fft(imstack, qarr):
	rows, cols, time = imstack.shape
	px_ts = []
	rclist = []

	# extract pixelwise timeseries
	for row in range(rows):
		for col in range(cols):
			ts_arr = imstack[row,col,:]

			if not np.isnan(ts_arr).all():
				px_ts.append(pd.Series(ts_arr))
				rclist.append([row,col])
			else:
				px_ts.append(pd.Series(np.zeros_like(ts_arr)))
				rclist.append([row,col])

	pxdf = pd.concat(px_ts, axis = 1)
	pxdf.columns = pxdf.columns.map(str)

	# Build the out image
	lagim = np.zeros_like(np.mean(imstack, axis = 2))
	corrim = np.zeros_like(np.mean(imstack, axis = 2))
	pvalim = np.zeros_like(np.mean(imstack, axis = 2))

	# Populate the per-pixel lags 
	for rc, dfcolidx in tqdm(list(zip(rclist,pxdf.columns))):

		a=pxdf[dfcolidx].values
		b=qarr.copy()

		# compute shift + corr mag

		# Shift
		try:
			A = fftpack.fft(a)
			B = fftpack.fft(b)
			Ar = -A.conjugate()
			shiftval = np.argmax(np.abs(fftpack.ifft(Ar*B)))
		except:
			shiftval = np.nan

		try:
			corrcoef = stats.pearsonr(a,b)
			corr = corrcoef[0]
			pval = corrcoef[1]
		except:
			pval = np.nan
			corr = np.nan

		# fill ims
		rowidx, colidx = rc
		lagim[rowidx,colidx] = shiftval
		corrim[rowidx,colidx] = abs(corr)
		pvalim[rowidx,colidx] = pval

	return lagim.astype(float), corrim.astype(float), pvalim.astype(float)

def fft_wrapper(imstack,qarr):

	lag_im, corr_im, pval_im = calc_xcorr_fft(imstack,qarr)

	# Get sum of stack 
	im_sum = np.nansum(imstack, axis = 2)
	
	# Mask where sum is zero
	lag_im, corr_im = [np.where(im_sum==0, np.nan, x) for x in [lag_im,corr_im]]

	# Filter lag and cor by >0.001 mm tottalthreshold for smlt and >1 for precip 
#     lag_im, corr_im = [np.where(im_mean<np.nanpercentile(im_mean,5), np.nan, x) for x in [lag_im,corr_im]]

	# Filter lag and cor by P value >0.05
#     lag_im, corr_im = [np.where(pval_im>0.05, np.nan, x) for x in [lag_im,corr_im]]
	
	# Mask nans from the pval image 
	lag_im = np.where(np.isnan(pval_im), np.nan, lag_im)

	return lag_im, corr_im

def mi_wrapper(imstack,qarr):

	mi_im = calc_mi(imstack,qarr)

	# Get mean of theentire stack 
	im_sum = np.nansum(imstack, axis = 2)

	# mask where sum is zero 
	mi_im = np.where(im_sum==0, np.nan, mi_im)

	# Mask the other nans 
	mi_im = np.where(np.isnan(im_sum), np.nan, mi_im)

	return mi_im

def main(stn_id):

	print("=======" * 15)
	print("PROCESSING: {}".format(stn_id))
	print("=======" * 15)

	# study period 
	dt_idx = pd.date_range('2003-10-01','2022-09-30', freq='D')

	# Read watershed gdf
	stn_gdf = gp.read_file("../shape/{}.shp".format(stn_id))

	# Read runoff
	rdf = pd.read_csv("../data/CDEC/runoff.csv")
	rdf['date'] = pd.to_datetime(rdf['date'])
	rdf.set_index("date", inplace = True)    

	smlt_fn_1d = "../data/Watersheds/1d/{}_1d_smlt.npy".format(stn_id)
	prcp_fn_1d = "../data/Watersheds/1d/{}_1d_prcp.npy".format(stn_id)

	smlt_fn_3d = "../data/Watersheds/3d/{}_3d_smlt.npy".format(stn_id)
	prcp_fn_3d = "../data/Watersheds/3d/{}_3d_prcp.npy".format(stn_id)

	smlt_fn_5d = "../data/Watersheds/5d/{}_5d_smlt.npy".format(stn_id)
	prcp_fn_5d = "../data/Watersheds/5d/{}_5d_prcp.npy".format(stn_id)

	# Load arrays 
	smlt = np.load(smlt_fn_1d)
	prcp = np.load(prcp_fn_1d)

	smlt_3d = np.load(smlt_fn_3d)
	prcp_3d = np.load(prcp_fn_3d)

	smlt_5d = np.load(smlt_fn_5d)
	prcp_5d = np.load(prcp_fn_5d)

	# if station is PAR, we need to chop off 2017 - 2021
	if stn_id == "PAR":
	    df = rdf[stn_id]
	    mask = (df.index <= "2016-09-30") 
	    stn_r_df = df.loc[mask].interpolate(how = 'linear')
	    n_days = len(stn_r_df)
	    smlt = smlt[:,:,:n_days]
	    prcp = prcp[:,:,:n_days]

	    smlt_3d = smlt_3d[:,:,:n_days]
	    prcp_3d = prcp_3d[:,:,:n_days]

	    smlt_5d = smlt_5d[:,:,:n_days]
	    prcp_5d = prcp_5d[:,:,:n_days]
	    years = range(2003,2017)

	else:
	    stn_r_df = rdf[stn_id].interpolate(how = 'linear')
	    years = range(2003,2022)

	# read runoff as array
	qarr = stn_r_df.values

	# copy the runoff df to modify
	shed_ts = pd.DataFrame(stn_r_df)

	# Make a watershed mean df
	shed_ts  = pd.DataFrame(stn_r_df.loc[dt_idx[0]:dt_idx[-1]])
	smlt_vals = np.array([np.nansum(x) for x in [smlt[:,:,t] for t in range(0, smlt.shape[2])]])
	prcp_vals = np.array([np.nansum(x) for x in [prcp[:,:,t] for t in range(0, prcp.shape[2])]])

	shed_ts['prcp_1d'] = prcp_vals
	shed_ts['smlt_1d'] = smlt_vals

	# calc watershed avg rolling sums
	shed_ts['prcp_3d'] = shed_ts['prcp_1d'].rolling(3).sum()
	shed_ts['smlt_3d'] = shed_ts['smlt_1d'].rolling(3).sum()

	shed_ts['prcp_5d'] = shed_ts['prcp_1d'].rolling(5).sum()
	shed_ts['smlt_5d'] = shed_ts['smlt_1d'].rolling(5).sum()

	########### Main Routine: Info Theory #############

	############## Extremes of 1,3,5d #############

	# for the extremes (1d, 3d, 5d), find the dates of the events that exceed the 99th %ile, and rank based on value for N events. 
	num_events = 5

	p_ev_1d = shed_ts[shed_ts['prcp_1d']>= np.nanpercentile(shed_ts['prcp_1d'], 99)].sort_values(by = 'prcp_1d', ascending = True).index[:num_events]
	d_ev_1d = shed_ts[shed_ts['smlt_1d']>= np.nanpercentile(shed_ts['smlt_1d'], 99)].sort_values(by = 'smlt_1d', ascending = True).index[:num_events]

	p_ev_3d = shed_ts[shed_ts['prcp_3d']>= np.nanpercentile(shed_ts['prcp_3d'], 99)].sort_values(by = 'prcp_3d', ascending = True).index[:num_events]
	d_ev_3d = shed_ts[shed_ts['smlt_3d']>= np.nanpercentile(shed_ts['smlt_3d'], 99)].sort_values(by = 'smlt_3d', ascending = True).index[:num_events]

	p_ev_5d = shed_ts[shed_ts['prcp_5d']>= np.nanpercentile(shed_ts['prcp_5d'], 99)].sort_values(by = 'prcp_5d', ascending = True).index[:num_events]
	d_ev_5d = shed_ts[shed_ts['smlt_5d']>= np.nanpercentile(shed_ts['smlt_5d'], 99)].sort_values(by = 'smlt_5d', ascending = True).index[:num_events]

	############# MAIN ROUTINE #############
	imstack_lookup = {
	'prcp_1d': prcp,
	'prcp_3d': prcp_3d,
	'prcp_5d': prcp_5d,
	'smlt_1d': smlt,
	'smlt_3d': smlt_3d,
	'smlt_5d': smlt_5d}

	# Zip together the dates we defined by thresholding with the keys to the relevant arrays 
	iterdict = dict(zip(list(imstack_lookup.keys()),[p_ev_1d,p_ev_3d,p_ev_5d,d_ev_1d,d_ev_3d,d_ev_5d]))

	for k,v in iterdict.items():
	    vartype = k
	    dates = v
	    imstack = imstack_lookup[k]

	    print("PROCESSING {}".format(k))
	    # Wrappers for main routines 
	    for start_date in dates:
	        
	        date_str = start_date.strftime('%Y_%m_%d')
	        
	        dt_idx = stn_r_df.index
	        timespan = start_date + relativedelta(months = 3) #  timespan after event
	        window = (dt_idx[dt_idx >= start_date]& dt_idx[dt_idx <= timespan])
	        
	        # Copy the df for indices to filter the array
	        ts = shed_ts.copy()
	        ts['dt'] = ts.index
	        ts.reset_index(inplace = True)
	        
	        start = ts[ts.dt == window[0]].index
	        end = ts[ts.dt == window[-1]].index

	        s, e = int(start.values), int(end.values)
	        win_stack = imstack[:,:,s:e+1]
	        qarr_in = shed_ts.loc[window][stn_id].interpolate(how = 'linear').values
	        
	        sum_fn = "../results/extremes/{}_{}_{}_{}.tif".format(stn_id,vartype,date_str,"sum")
	        mi_fn = "../results/extremes/{}_{}_{}_{}.tif".format(stn_id,vartype,date_str,"mi")
	        lag_fn = "../results/extremes/{}_{}_{}_{}.tif".format(stn_id,vartype,date_str,"lag")
	        cor_fn = "../results/extremes/{}_{}_{}_{}.tif".format(stn_id,vartype,date_str,"cor")
	        
	        # Check if outfiles already exist 
	        filelist = [sum_fn, mi_fn, lag_fn, cor_fn]

	        # If they don't process and write
	        if not all([os.path.isfile(f) for f in filelist]):

	            im_sum = np.nansum(win_stack, axis = 2)
	            lag, corr = fft_wrapper(win_stack, qarr_in)
	            mi = mi_wrapper(win_stack, qarr_in)

	            write_raster(lag, stn_gdf, lag_fn)
	            write_raster(corr, stn_gdf, cor_fn)
	            write_raster(mi, stn_gdf, mi_fn)
	            write_raster(im_sum, stn_gdf, sum_fn)

if __name__ == "__main__":

	# Read watersheds
	gdf = gp.read_file("../shape/sierra_catchments.shp")

	stids_all = list(gdf['stid'].values)
	nodata_stids = ["MCR", "CFW"]

	stids = [x for x in stids_all if x not in nodata_stids][::-1]

	for stid in stids:
	  main(stid)

	# pool = mp.Pool(2)
	# for i in tqdm(pool.imap_unordered(main, stids), total=len(stids)):
	# 	pass