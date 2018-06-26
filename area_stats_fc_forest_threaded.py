
# Identify areas likely to be forest based on Green Cover threshold
# Multi-threaded.
# Calls gdal_merge.py to merge images when an area is too big and polygons are analyzed individually.

import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import collections
import pickle

import rasterio
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

def fc_p10_file(year):
    return '../FC/p10/FC_%d_10.vrt'%year

BAND=1 #depend on FC file type

#!!! CONFIRM period for analysis!!!#
#for 2006 to 2010
FC_YEARS_PROJECT=range(2006,2011)

savefig=True
FIGS_PATH='figs_forest_%d_%d'%(FC_YEARS_PROJECT[0],FC_YEARS_PROJECT[-1])

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
    
def fc_b_worker(iarea, gdf, fc_years_before=range(2006,2011), band=0, savefig=False, figs_path='figs'):
    """
    worker thread function
    """
    area_geoms=gdf.geometry.values[iarea]
    split=False
    if area_geoms.envelope.area>1e10:split=True #area is too big

    thresh=30 # above which FPC <0.1 is not likely
    npixels_tot=0
    forest_pixels=0
    if not split: area_geoms=[area_geoms]
    
    # save rasters
    if savefig:
        outputdir='{0}/{1}'.format(figs_path,gdf['PROJ_ID'][iarea])
        if not os.path.exists(outputdir): os.mkdir(outputdir)
        figname='{0}/{1}_forest.tif'.format(outputdir,gdf['PROJ_ID'][iarea])

    for idx, area_geom in enumerate(area_geoms):
        npixels=0
        forest_mask=None
        for year in fc_years_before:
            area_image,area_transform = pixels_of_interest(fc_p10_file(year),[area_geom], band=band)
            if npixels==0:
                npixels=area_image.count()
            if npixels==0:break
            if forest_mask is None:
                forest_mask=area_image>thresh
            else:
                forest_mask=np.logical_or(forest_mask,area_image>thresh)
        if npixels==0:continue
        npixels_tot+=npixels
        forest_pixels+=forest_mask.sum()

        # save rasters
        if savefig:
            if split: 
                figname_area=figname.replace('.tif', '_{0:05d}.tif'.format(idx))
            else: 
                figname_area = figname
            with rasterio.open(figname_area,
                               'w',driver='GTiff',
                               width=forest_mask.shape[1],height=forest_mask.shape[0],
                               dtype=np.int16, count=1, nodata=NODATA,
                               transform=area_transform,
                               crs={'init': 'epsg:%d'%EPSG}) as dst:
                dst.write((1.*forest_mask).filled(NODATA).astype(np.int16),1)
                
    if savefig and split:
        # combine the figures into one
        inputlist=figname.replace('.tif','_?????.tif')
        #this will call gdal_merge
        subprocess.call('gdal_merge.py -init {0} -n {0} -a_nodata {0} -o {1} {2}'.format(
                NODATA,figname,inputlist), shell=True)
    
    forest_percent=forest_pixels*100./npixels_tot
    print("# of pixels loaded for project %d, %s:"%(iarea,gdf['PROJ_ID'][iarea]),npixels_tot)
    return (gdf['PROJ_ID'][iarea], npixels_tot,forest_pixels,forest_percent)

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
csvname=projectfile.split('/')[-1].replace('.shp','_forest_%d_%d.csv'%(FC_YEARS_PROJECT[0],FC_YEARS_PROJECT[-1]))

#pickling to make sure result is saved where possible;
#and if interrupted, can run from last saved point.
picklename= projectfile.split('/')[-1].replace('.shp','_forest_%d_%d.pkl'%(FC_YEARS_PROJECT[0],FC_YEARS_PROJECT[-1]))
if os.path.exists(picklename): df=pickle.load(open(picklename,'rb'))
else:
    fc_output=collections.OrderedDict()
    fc_output['PROJ_ID']=[]
    fc_output['npixels']=[]
    fc_output['forest_pixels']=[]
    fc_output['forest_percent']=[]
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
fc_b_func=partial(fc_b_worker, gdf=gdf, savefig=savefig,
                  fc_years_before=FC_YEARS_PROJECT, figs_path=FIGS_PATH, 
                  band=BAND)  
for result in pool.imap(fc_b_func, idx):
    proj_id, npixels, forest_pixels, forest_percent = result
    
    df=df.append({'PROJ_ID':proj_id,'npixels':npixels,
                  'forest_pixels':forest_pixels,
                  'forest_percent':forest_percent},
                 ignore_index=True)
    pickle.dump(df, open(picklename,'wb'))

# finish for this job
pool.close()
pool.join()
df.to_csv(csvname, float_format='%.1f')
