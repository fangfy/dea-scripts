
# extract stats for areas in a shapefile

import os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import collections

import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

def pixels_of_interest(raster_path, geoms, band=None):
    """
    Extract pixels within polygons from a raster.
    Geometries need to be in the same projection as the raster.
    """
    with rasterio.open(raster_path) as src:
        input_g=[mapping(g) for g in geoms]
        out_image, out_transform = mask(src, input_g, crop=True, filled=False, pad=True)
    if band:
        return out_image[band]
    else: return out_image


def main(projectfile, outputfile, wofs_summary,
         wofs_epsg=3577, wofs_thresh=[1,5,30]):
    """
    Extract wofs stats
    """
    #set up table for output
    wofs_output=collections.OrderedDict()
    wofs_output['PROJ_ID']=[]
    for thresh in wofs_thresh:
        wofs_output['wofs_%d'%thresh]=[]
        
    #load project shape file
    gdf=gpd.read_file(projectfile)
    gdf=gdf.to_crs(epsg=wofs_epsg)
    narea=len(gdf.index)
    
    #loop through project areas
    for iarea in range(narea):
        print("AREA:",iarea,gdf['PROJ_ID'][iarea])
        area_geom=gdf.geometry.values[iarea]

        wofs_output['PROJ_ID'].append(gdf['PROJ_ID'][iarea])
        area_image =pixels_of_interest(wofs_summary, [area_geom])
        for thresh in wofs_thresh:
            wofs_percent=(area_image>thresh).sum()*100./area_image.count()
            print("Percentage of pixels with >%d water detection:"%thresh,
                  round(wofs_percent))
            wofs_output['wofs_%d'%thresh].append(wofs_percent)
        
    df=pd.DataFrame(data=wofs_output)
    df.to_csv(outputfile,float_format='%.1f')


if len(sys.argv)<2:
    print("Please provide master shape file in the command line.")
    exit()

projectfile=sys.argv[1]
outputfile=projectfile.split('/')[-1].replace('.shp','_wofs.csv')

wofs_summary='WOFS/confidence_filtered10.vrt'

print("Generating stats for areas in", projectfile)
print("Using WOFS summary", wofs_summary)
print("Outputing to", outputfile)
main(projectfile,outputfile,wofs_summary)
