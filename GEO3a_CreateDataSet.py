import os
from glob import glob # File manipulation
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import pandas as pd
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import time
import sys
from rio_toa import toa_utils
from rio_toa import sun_utils
import l8qa.qa as l8QA


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------------------------------------------------------------------------------------------------
from GEOUtil_CommonUtils import GetFilePath
from GEOUtil_CommonUtils import GetImagesPath
file_path = GetFilePath()
image_path = GetImagesPath()
iloscP = 10
skok = 5.0
os.chdir(file_path)
os.listdir(file_path)

# -------------------------------------------------------------------------------------------------------
df = pd.read_csv(file_path + '1_DataExtract/imageList.csv', sep = ';')
searchRec = pd.read_hdf(file_path + '1_DataExtract/searchRec.h5', 'searchRec')
print('{}\n'.format(df))

# -------------------------------------------------------------------------------------------------------
fileExt = ['_B1.TIF', '_B2.TIF', '_B3.TIF', '_B4.TIF', '_B5.TIF', '_B6.TIF', '_B7.TIF', '_B8.TIF', '_B9.TIF', '_B10.TIF', '_B11.TIF', '_BQA.TIF', '_MTL.txt', '_ANG.txt']
bands = fileExt[:12]
dataset = pd.DataFrame(); #columns=['image_id','AirQualityStation','Latitude','Longitude','North','East', 'Cloud'])
pollutant = []

# -------------------------------------------------------------------------------------------------------
def create_spiral(dfTemp, band, k, l, i_row, i_col, band_in, incr):
    k = k+1
    if k <= iloscP:
        x_incr = k%2
        y_incr = (k+1)%2
        j = k//2
        i_row = int(round(i_row + j*pow(-1, j) * (x_incr * incr)))
        i_col = int(round(i_col + j*pow(-1, j) * (y_incr * incr)))
        if band != '_BQA.TIF':
            try:
                value = band_in[i_row, i_col]
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp[band[1:][:-4] + '_' + str(k)] = value
        else:
            # https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
            try:
                value = l8QA.fill_qa(band_in[i_row, i_col])                  # Designated Fill: Areas which are not imaged or could not be populated with data, but are part of the image’s grid
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp['QA_fill'] = value
            try:
                value = l8QA.terrain_qa(band_in[i_row, i_col])               # Dropped Pixel: An error has occurred during acquisition and erroneous data have been populated into this pixel
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp['QA_terrain'] = value
            try:
                value = l8QA.radiometric_qa(band_in[i_row, i_col])           # Radiometric Saturation: Indicates how many bands contain saturation
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp['QA_radiometric'] = value
            try:
                value = l8QA.cloud_confidence(band_in[i_row, i_col])          # Cloud Confidence: Confidence in the pixel containing any type of cloud
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp['QA_cloudConf'] = value
            try:
                value = l8QA.cloud(band_in[i_row, i_col])                    # Cloud: Indicates whether or not the pixel contains cloud
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp['QA_cloud'] = value
            try:
                value = l8QA.cloud_shadow_confidence(band_in[i_row, i_col])  # Cloud Shadow Confidence: Confidence in the pixel containing cloud shadow
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp['QA_cloudShadow'] = value
            try:
                value = l8QA.snow_ice_confidence(band_in[i_row, i_col])      # Snow/Ice Confidence: Confidence in the pixel containing snow and/or ice
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp['QA_snowIce'] = value
            try:
                value = l8QA.cirrus_confidence(band_in[i_row, i_col])        #  Cirrus Confidence: Confidence in the pixel containing cirrus cloud
            except:
                print("Unexpected error:", sys.exc_info()[0])
                value = -1
            dfTemp['QA_cirrusConf'] = value
        # next iteration of recurance        
        dfTemp = create_spiral(dfTemp, band, k, l, i_row, i_col, band_in, incr)
    return dfTemp

def processBand(band, dfTemp, imgWidth):
    lon,lat =  float(dfTemp['Longitude'][0]), float(dfTemp['Latitude'][0])
    srcFile = image_path + dfTemp['image_id'][0] + band
    if os.path.isfile(srcFile):
        print(str(i).rjust(5,' '), '/', str(df.shape[0]).rjust(5,' '), time.strftime("  -> %d.%m.%Y %H:%M:%S"), dfTemp['image_id'][0] + band)
        # -------------- Band data
        bandImg = rasterio.open(srcFile)
        if imgWidth == 0:
            imgWidth = bandImg.width
        # https://geohackweek.github.io/raster/04-workingwithrasters/
        # https://www.movable-type.co.uk/scripts/latlong-utm-mgrs.html
        # STA.DE_DEUB004: east, noth	418408.3600937926, 5307236.104232612  lat,lon      47.91325599999999, 7.908035
        # STA.DE_DEHE051: east, noth	566374.975363431, 5594389.15693539    lat,lon      50.497711, 9.935862
        utm = pyproj.Proj(bandImg.crs) # Pass CRS of image from rasterio
        lonlat = pyproj.Proj(init='epsg:4326')
        east,north = pyproj.transform(lonlat, utm, lon, lat)
        dfTemp['North'] = north
        dfTemp['East'] = east
        #transg  = Transformer.from_crs("epsg:4326", band.crs)
        #e1, n1 = transg .transform(lon, lat)
        i_row, i_col = bandImg.index(east, north)
        k = 0
        l = 0
        band_in = bandImg.read(1)
        dfTemp = create_spiral(dfTemp, band, k, l, i_row, i_col, band_in, skok * bandImg.width / imgWidth)
    return dfTemp, imgWidth


# -------------------------------------------------------------------------------------------------------
def processBandSet(i, row, dfTemp):
    # metadata
    mtl = toa_utils._load_mtl(image_path + dfTemp['image_id'][0] + '_MTL.txt')
    metadata = mtl['L1_METADATA_FILE']
    for b in range(1,10,1):
        dfTemp['REFLECTANCE_MULT_BAND_{}'.format(b)] = metadata['RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_{}'.format(b)] 
        dfTemp['REFLECTANCE_ADD_BAND_{}'.format(b)]  = metadata['RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_{}'.format(b)]
    for b in range(10,12,1):
        dfTemp['K1_CONSTANT_BAND_{}'.format(b)] = metadata['TIRS_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_{}'.format(b)] 
        dfTemp['K2_CONSTANT_BAND_{}'.format(b)] = metadata['TIRS_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_{}'.format(b)]
        dfTemp['RADIANCE_MULT_BAND_{}'.format(b)] = metadata['RADIOMETRIC_RESCALING']['RADIANCE_MULT_BAND_{}'.format(b)]
        dfTemp['RADIANCE_ADD_BAND_{}'.format(b)] = metadata['RADIOMETRIC_RESCALING']['RADIANCE_ADD_BAND_{}'.format(b)]
    dfTemp ['SUN_ELEVATION'] = metadata['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
    dfTemp ['DATE_ACQUIRED'] = date_collected = metadata['PRODUCT_METADATA']['DATE_ACQUIRED']
    dfTemp ['SCENE_CENTER_TIME']= metadata['PRODUCT_METADATA']['SCENE_CENTER_TIME']

    # poluntant columns
    for elem in pollutant:
        dfTemp[elem] = row[elem]
        
    # band data
    imgWidth = 0
    for bandNo in bands:
        dfTemp, imgWidth = processBand(bandNo, dfTemp, imgWidth)
    return dfTemp

# -------------------------------------------------------------------------------------------------------
# select polutantas
for row in df.columns[7:].values:
    if (df[row].isnull().sum(axis=0)/df.shape[0] <= 0.4):
        pollutant.append(row)
print(pollutant)

# -------------------------------------------------------------------------------------------------------
for i,row in df.iterrows():
    image_id = row['displayId']
    dfTemp = pd.DataFrame([{'image_id': image_id, 'AirQualityStation': row['AirQualityStation'], 'Latitude' : row['Latitude'], 'North' : '', 'East' : '', 'Longitude' : row['Longitude'], 'Cloud' : row['cloudCover']}])

    numFi = 0
    for fe in fileExt:
        if os.path.isfile(image_path + image_id + fe):
            numFi += 1

    if numFi == len(fileExt):
        dfTemp = processBandSet(i, row, dfTemp)
        # -------------- add and save
        dataset = dataset.append(dfTemp)
        dataset.to_csv(file_path + '3_CreatingDataset/data_set_a.csv', index = False, header=True, sep = ';')
    else:
        print(str(i).rjust(5,' '), '/', str(df.shape[0]).rjust(5,' '), time.strftime("  -> %d.%m.%Y %H:%M:%S"), ' !!! skiping {} - number of images only {}'.format(image_id, numFi))
