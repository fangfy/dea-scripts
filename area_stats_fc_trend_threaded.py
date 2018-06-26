
# Analyze Fractional Cover trend for areas in a shapefile
# Multi-threaded.
# Calls gdal_merge.py to merge images when an area is too big and polygons are analyzed individually.

import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
import collections
import pickle

import rasterio
from rasterio.enums import ColorInterp
from rasterio.mask import mask
from functools import partial
from shapely.geometry import mapping

import multiprocessing
from multiprocessing.pool import ThreadPool as Pool

import subprocess

########################################
NODATA=-999
EPSG=3577

#!!! CONFIRM location of FC files!!!#
FC_PATH='../../COG/Geotiff-conversion/FC_PC'

# where to find the FC files
def fc_pv_p10_file(year, fc_path=FC_PATH):
    return '%s/FC_PV_PC_10_%d.vrt'%(fc_path,year)

BAND=0 # this depend on FC file type

#!!! CONFIRM period for trend analysis!!!#
#for 2012 to 2016
FC_YEARS_PROJECT=range(2012,2017)
#for 2013 to 2017
#FC_YEARS_PROJECT=range(2013,2018)

savefig=True
FIGS_PATH='figs_%d_%d'%(FC_YEARS_PROJECT[0],FC_YEARS_PROJECT[-1])

#maximum number of threads to use
max_nworker=4
#########################################

def pixels_of_interest(raster_path, geoms, band=1):
    with rasterio.open(raster_path) as src:
        input_g=[mapping(g) for g in geoms]
        out_image, out_transform = mask(src, input_g, crop=True, filled=False, pad=True)
    if band is None:
        return out_image, out_transform
    else:
        return out_image[band], out_transform
    
def linear_fit(Y,X,novalue=NODATA):
    """pixel-by-pixel ordinary least square fit for a trend.
    returns slope and pvalue for slope==0
    """
    if len(Y.compressed())==0: return [novalue, novalue]
    model=sm.OLS(Y,X,missing='drop').fit()
    if model.params[1]==0: return [0., 1.]
    return [model.params[1], model.pvalues[1]]

def fc_p_worker(iarea,gdf,savefig=False, fc_years_project=range(2012,2017), figs_path='figs', band=0):
    area_geoms=gdf.geometry.values[iarea]
    split=False
    if area_geoms.envelope.area>1e9:split=True #area is too big to do at once

    npixels_tot=0
    negative_pixels=0
    negative_mean=0.
    negative_slope=0.
    positive_pixels=0
    positive_mean=0.
    positive_slope=0.
    if not split: area_geoms=[area_geoms]

    # save rasters
    if savefig:
        outputdir='{0}/{1}'.format(figs_path,gdf['PROJ_ID'][iarea])
        if not os.path.exists(outputdir): os.mkdir(outputdir)
        figname='{0}/{1}_fc_pvtrend_masked.tif'.format(outputdir,gdf['PROJ_ID'][iarea])
        
    for idx,area_geom in enumerate(area_geoms):
        if area_geom.area<90: continue  #skip areas that are too small and may be impacted by edge effect
        npixels=0
        pv_stack=None
        x=np.array(fc_years_project)
        # stack PV values
        for year in fc_years_project:
            pv, pv_transform =pixels_of_interest(fc_pv_p10_file(year), [area_geom], band=band)
            if npixels==0:
                npixels=pv.count()
            if npixels==0:break
            if pv_stack is None: pv_stack=pv
            else: pv_stack=np.ma.dstack((pv_stack,pv))

        if npixels==0: continue
        pv_mean=pv_stack.mean(axis=2)
        
        # fit trend
        X=sm.add_constant(x)
        pv_peryear=np.ma.apply_along_axis(linear_fit,axis=2,arr=pv_stack, X=X, novalue=NODATA)
        pv_peryear=np.ma.masked_values(pv_peryear, NODATA)
        pvalue=pv_peryear[:,:,1]
        pv_peryear=pv_peryear[:,:,0]
        
        # mask negative and positive based on tstat
        npixels_tot+=npixels
        negative_mask=(pv_peryear<0) & (pvalue<0.1)
        n_negative=negative_mask.sum()
        positive_mask=(pv_peryear>0) & (pvalue<0.1)
        n_positive=positive_mask.sum()
        
        if n_negative>0:
            negative_pixels+=n_negative
            negative_mean+=pv_mean[negative_mask].sum()
            negative_slope+=pv_peryear[negative_mask].sum()
            
        if n_positive>0:
            positive_pixels+=n_positive
            positive_mean+=pv_mean[positive_mask].sum()
            positive_slope+=pv_peryear[positive_mask].sum()

        # save rasters
        if savefig:
            if split: 
                figname_area=figname.replace('.tif', '_{0:05d}.tif'.format(idx))
            else: 
                figname_area = figname
            with rasterio.open(figname_area,
                               'w',driver='GTiff',
                               width=pv_peryear.shape[1],height=pv_peryear.shape[0],
                               dtype=np.int32, count=3, nodata=NODATA,
                               transform=pv_transform,
                               crs={'init': 'epsg:%d'%EPSG}) as dst:
                dst.colorinterp = (
                    ColorInterp.red, ColorInterp.green, ColorInterp.blue,
                    )
                dst.write(np.ma.masked_array(1*negative_mask,
                                             mask=pv_peryear.mask).filled(NODATA).astype(np.int32),1)
                dst.write(pv_mean.filled(NODATA).astype(np.int32),2)
                dst.write(np.ma.masked_array(1*positive_mask,
                                             mask=pv_peryear.mask).filled(NODATA).astype(np.int32),3) 
                dst.set_description(1, "Band 1 Negative Trend Mask")
                dst.set_description(2, "Band 2 Mean PV")
                dst.set_description(3, "Band 3 Positive Trend Mask")
            ## It's possible to use a different band combination or include more bands,
            ## but it's easiest to work with three-band RGB (e.g. this doesn't break gdal_merge).
            ## Another example with RGB-apha:
            #with rasterio.open(figname2_area,
            #                   'w',driver='GTiff',
            #                   width=pv_peryear.shape[1],height=pv_peryear.shape[0],
            #                   dtype=pv_peryear.dtype, count=4, nodata=NODATA,
            #                   transform=pv_transform,
            #                   crs={'init': 'epsg:%d'%EPSG}) as dst:
            #    dst.colorinterp = (
            #        ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha, 
            #        )
            #    dst.write(np.ma.masked_where(pv_peryear>0, (-1*pv_peryear).filled(NODATA)).filled(0),1)
            #    dst.write(pv_mean.filled(NODATA),2)
            #    dst.write(np.ma.masked_where(pv_peryear<0, pv_peryear.filled(NODATA)).filled(0),3)
            #    dst.write(255*(1-pvalue).filled(1),4)
            #    dst.set_description(1, "Band 1 Negative Trend")
            #    dst.set_description(2, "Band 2 Mean PV")
            #    dst.set_description(3, "Band 3 Positive Trend")
            #    dst.set_description(4, "Band 4 255*(1-pvalue)")


    if savefig and split:
        # combine the figures into one
        inputlist=figname.replace('.tif','_?????.tif')
        #this will call gdal_merge
        subprocess.call('gdal_merge.py -init {0} -n {0} -a_nodata {0} -o {1} {2}'.format(
                NODATA,figname,inputlist), shell=True)
            
    negative_percent=negative_pixels*100./npixels_tot
    positive_percent=positive_pixels*100./npixels_tot
    if negative_pixels>0:
        negative_mean/=negative_pixels
        negative_slope/=negative_pixels
    if positive_pixels>0:
        positive_mean/=positive_pixels
        positive_slope/=positive_pixels

    #print out results
    print(gdf['PROJ_ID'][iarea], npixels_tot,negative_pixels,negative_percent,negative_mean,negative_slope, positive_pixels, positive_percent, positive_mean, positive_slope)
    return (gdf['PROJ_ID'][iarea], npixels_tot,negative_pixels,negative_percent,
            negative_mean,negative_slope, 
            positive_pixels, positive_percent, positive_mean, positive_slope)


#########################################

if len(sys.argv)<2:
    print("Please provide master shape file in the command line.")
    exit()

projectfile=sys.argv[1]

gdf=gpd.read_file(projectfile)
gdf=gdf.to_crs(epsg=EPSG)
narea=len(gdf.index)

#start the workers
ncpus=multiprocessing.cpu_count()
nworkers=min(ncpus, max_nworker)
print("# of works started:",nworkers)
pool = Pool(nworkers)

#output spreadsheet
csvname=projectfile.split('/')[-1].replace('.shp','_fc_trend_%d_%d.csv'%(FC_YEARS_PROJECT[0],FC_YEARS_PROJECT[-1]))

#pickling to make sure result is saved where possible;
#and if interrupted, can run from last saved point.
picklename= projectfile.split('/')[-1].replace('.shp','_fc_trend_%d_%d.pkl'%(FC_YEARS_PROJECT[0],FC_YEARS_PROJECT[-1]))
if os.path.exists(picklename): df=pickle.load(open(picklename,'rb'))
else:
    fc_output=collections.OrderedDict()
    fc_output['PROJ_ID']=[]
    fc_output['npixels']=[]
    fc_output['npixels_with_negative_trend']=[]
    fc_output['percent_with_negative_trend']=[]
    fc_output['mean_PV_for_negative_area']=[]
    fc_output['mean_trend_for_negative_area']=[]
    fc_output['npixels_with_positive_trend']=[]
    fc_output['percent_with_positive_trend']=[]
    fc_output['mean_PV_for_positive_area']=[]
    fc_output['mean_trend_for_positive_area']=[]
    df=pd.DataFrame(fc_output)


if len(df['PROJ_ID'])>0:
    # pick up from last saved point, assuming projects are in the same order
    for iarea in range(narea):
        if any(df['PROJ_ID']==gdf['PROJ_ID'][iarea]): continue
        break
    idx=range(iarea,narea)
else:
    idx=range(narea)

if savefig and (not os.path.exists(FIGS_PATH)): os.mkdir(FIGS_PATH)
fc_p_func=partial(fc_p_worker, gdf=gdf, savefig=savefig,
                  fc_years_project=FC_YEARS_PROJECT, figs_path=FIGS_PATH, band=BAND)  
for result in pool.imap(fc_p_func, idx):
    proj_id, npixels, negative_pixels, negative_percent, negative_mean, negative_slope, \
        positive_pixels, positive_percent, positive_mean, positive_slope \
        =result

    df=df.append({'PROJ_ID':proj_id,'npixels':npixels,
                  'npixels_with_negative_trend':negative_pixels,
                  'percent_with_negative_trend':negative_percent,
                  'mean_PV_for_negative_area':negative_mean,
                  'mean_trend_for_negative_area':negative_slope,
                  'npixels_with_positive_trend':positive_pixels,
                  'percent_with_positive_trend':positive_percent,
                  'mean_PV_for_positive_area':positive_mean,
                  'mean_trend_for_positive_area':positive_slope},
                 ignore_index=True)
    pickle.dump(df, open(picklename,'wb'))

# finish for this job
pool.close()
pool.join()
df.to_csv(csvname, float_format='%.1f')
