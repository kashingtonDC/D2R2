import os

import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gp

from osgeo import gdal, osr
from tqdm import tqdm
from sklearn import metrics
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

	# Read rainfall and snowmelt data
	smlt_fn_1d = "../data/Watersheds/1d/{}_1d_smlt.npy".format(stn_id)
	prcp_fn_1d = "../data/Watersheds/1d/{}_1d_prcp.npy".format(stn_id)

	smlt = np.load(smlt_fn_1d)
	prcp = np.load(prcp_fn_1d)

	# filter the runoff data to select watershed

	# Set time range

	# if station is PAR, we need to chop off 2017 - 2021
	if stn_id == "PAR":
	    df = rdf[stn_id]
	    mask = (df.index <= "2016-09-30") 
	    stn_r_df = df.loc[mask].interpolate(how = 'linear')
	    n_days = len(stn_r_df)
	    smlt = smlt[:,:,:n_days]
	    prcp = prcp[:,:,:n_days]
	    years = range(2003,2017)

	else:
	    stn_r_df = rdf[stn_id].interpolate(how = 'linear')
	    years = range(2003,2022)

	# read runoff as array
	qarr = stn_r_df.values

	# copy the runoff df to modify
	shed_ts = pd.DataFrame(stn_r_df)

	# Assign seasons to months
	shed_ts['month'] = shed_ts.index.month
	seasons = {10:'F', 11:'F', 12:'F', 1:'W', 2:'W', 3:'W', 4:'Sp', 5:'Sp', 6:'Sp',7:'Su',8:'Su',9:'Su'}
	shed_ts['Season'] = shed_ts['month'].apply(lambda x: seasons[x])

	# Map season to hydro year position
	seas_2_hy = {"F": "1", "W": "2", "Sp":"3", "Su":"4"}

	for y in tqdm(list(years)[:]):
	    ydf = shed_ts[shed_ts.index.year == y]

	    for season in list(ydf.Season.unique()):
	        sdf = ydf[ydf.Season==season]

	        # Get starting and ending indices of that season and subset data 
	        t1 = sdf.index[0]
	        t2 = sdf.index[-1]
	        window = (dt_idx[dt_idx > t1]& dt_idx[dt_idx <= t2])

	        # Copy the df for indices to filter the array
	        ts = shed_ts.copy()
	        ts['dt'] = ts.index
	        ts.reset_index(inplace = True)
	        start = ts[ts.dt == window[0]].index
	        end = ts[ts.dt == window[-1]].index

	        s, e = int(start.values), int(end.values)

	        # sum the prcp and smlt during that season
	        prcp_sum = np.nansum(prcp[:,:,s:e+1], axis =2)
	        smlt_sum = np.nansum(smlt[:,:,s:e+1], axis =2)

	        # Calc lag, cor, MI 
	        print("==========" * 5 )
	        print(y, season)
	        print("==========" * 5 )
	        
	        qarr_in = shed_ts.loc[window][stn_id].interpolate(how = 'linear').values

	        # calc cor, lag 
	        smlt_lag, smlt_corr = fft_wrapper(smlt[:,:,s:e+1], qarr_in)
	        prcp_lag, prcp_corr = fft_wrapper(prcp[:,:,s:e+1], qarr_in)

	        # calc MI
	        prcp_mi = mi_wrapper(prcp[:,:,s:e+1], shed_ts.loc[window][stn_id].interpolate(how = 'linear'))
	        smlt_mi = mi_wrapper(smlt[:,:,s:e+1], shed_ts.loc[window][stn_id].interpolate(how = 'linear'))

	        # write out
	        seas2hy = {"F":"1","W":"2","Sp":"3","Su":"4"}
	        hy_idx = seas2hy[season]
	        if season == "F":
	            hy = y+1
	        else:
	            hy = y

	        if not os.path.exists("../results/seasons/timeseries"):
	            os.mkdir("../results/seasons/timeseries")

	        prcp_sum_fn = "../results/seasons/timeseries/{}_prcp_sum_{}_{}.tif".format(stn_id,hy_idx,str(hy))
	        prcp_mi_fn = "../results/seasons/timeseries/{}_prcp_mi_{}_{}.tif".format(stn_id,hy_idx,str(hy))
	        prcp_lag_fn = "../results/seasons/timeseries/{}_prcp_lag_{}_{}.tif".format(stn_id,hy_idx,str(hy))
	        prcp_cor_fn = "../results/seasons/timeseries/{}_prcp_cor_{}_{}.tif".format(stn_id,hy_idx,str(hy))

	        smlt_sum_fn = "../results/seasons/timeseries/{}_smlt_sum_{}_{}.tif".format(stn_id,hy_idx,str(hy))
	        smlt_mi_fn = "../results/seasons/timeseries/{}_smlt_mi_{}_{}.tif".format(stn_id,hy_idx,str(hy))
	        smlt_lag_fn = "../results/seasons/timeseries/{}_smlt_lag_{}_{}.tif".format(stn_id,hy_idx,str(hy))
	        smlt_cor_fn = "../results/seasons/timeseries/{}_smlt_cor_{}_{}.tif".format(stn_id,hy_idx,str(hy))

	        write_raster(smlt_lag, stn_gdf, smlt_lag_fn)
	        write_raster(smlt_corr, stn_gdf, smlt_cor_fn)
	        write_raster(smlt_mi, stn_gdf, smlt_mi_fn)
	        write_raster(smlt_sum, stn_gdf, smlt_sum_fn)

	        write_raster(prcp_lag, stn_gdf, prcp_lag_fn)
	        write_raster(prcp_corr, stn_gdf, prcp_cor_fn)
	        write_raster(prcp_mi, stn_gdf, prcp_mi_fn)
	        write_raster(prcp_sum, stn_gdf, prcp_sum_fn)

	print("COMPLETED {}".format(stn_id))


if __name__ == '__main__':

	# Read watersheds
	gdf = gp.read_file("../shape/sierra_catchments.shp")

	stids_all = list(gdf['stid'].values)
	nodata_stids = ["MCR", "CFW"]

	stids = [x for x in stids_all if x not in nodata_stids]

	for stid in stids:
		main(stid)