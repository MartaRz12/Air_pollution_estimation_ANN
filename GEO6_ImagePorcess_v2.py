import sys
import os
import numpy as np
from numpy import loadtxt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
import keras.backend as K
from rio_toa import toa_utils   #pip install -U rio_toa
from rio_toa import sun_utils   #pip install -U rio_toa
import l8qa.qa as l8QA          #pip install rio-l8qa
import datetime
import rasterio
from rasterio.enums import Resampling

from GEOUtil_CommonUtils import GetImagesPath
from GEOUtil_CommonUtils import GetFilePath
file_path = GetImagesPath()
    #displayId = 'LC08_L1TP_188025_20190301_20190309_01_T1' # PL_KAT
    #displayId = 'LC08_L1TP_187024_20200328_20200409_01_T1' # PL_WAW
#displayId = 'LC08_L1TP_190022_20190331_20190404_01_T1' # PL_GDA
    #displayId = 'LC08_L1TP_196025_20190410_20190422_01_T1' # DE
displayId = 'LC08_L1TP_188023_20200404_20200410_01_T1'
#displayId = 'LC08_L1GT_188024_20200506_20200509_01_T2'
#displayId = 'LC08_L1GT_188025_20200506_20200509_01_T2'
#displayId = 'LC08_L1TP_187024_20200413_20200422_01_T1'
#displayId = 'LC08_L1TP_188023_20190301_20190309_01_T1'
#displayId = 'LC08_L1TP_188023_20200404_20200410_01_T1'
#displayId = 'LC08_L1TP_188024_20200420_20200508_01_T1'
#displayId = 'LC08_L1TP_188025_20190317_20190325_01_T1'
#displayId = 'LC08_L1TP_188025_20200216_20200225_01_T1'
#displayId = 'LC08_L1TP_188025_20200303_20200314_01_T2'
#displayId = 'LC08_L1TP_188026_20200319_20200326_01_T1'
#displayId = 'LC08_L1TP_189023_20190324_20190403_01_T1'
#displayId = 'LC08_L1TP_190022_20190331_20190404_01_T1'
#displayId = 'LC08_L1TP_190022_20200504_20200509_01_T1'
#displayId = 'LC08_L1TP_191022_20190306_20190309_01_T1'
#displayId = 'LC08_L1TP_191022_20200308_20200314_01_T1'
#displayId = 'LC08_L1TP_191022_20200409_20200422_01_T1'


tStart = datetime.datetime.now().replace(microsecond=0)

for fdir in os.walk(file_path):
    my_file = os.path.join(fdir[0], displayId + '_B1.TIF')
    if os.path.isfile(my_file):
        file_path = fdir[0] + '/'
        break

models = [GetFilePath() + '3_CreatingDataset/model_v2_NO_mae_82_39.h5',
          GetFilePath() + '3_CreatingDataset/model_v2_NO2_mae_67_33.h5',
          GetFilePath() + '3_CreatingDataset/model_v2_NOX as NO2_mae_59_17.h5',
          GetFilePath() + '3_CreatingDataset/model_v2_PM10_mae_96_59.h5',
          GetFilePath() + '3_CreatingDataset/model_v2_SO2_mae_91_28.h5'
          ]
fileExt = ['_B1.TIF', '_B2.TIF', '_B3.TIF', '_B4.TIF', '_B5.TIF', '_B6.TIF', '_B7.TIF', '_B8.TIF', '_B9.TIF', '_B10.TIF', '_B11.TIF', '_BQA.TIF']

# -------------------------------------------------------------------------------------------------------
for model in models:
    fname = (os.path.split(model)[-1]).split('_')
    dst_file = GetFilePath() + '3_CreatingDataset/' + displayId + '_' + fname[1] + '_' + fname[2] + '_' + fname[3]
    logFile = open(dst_file + '_v2.log', "w+")


    src_data = []
    #rescaler = [8081, 7981]
    rescaler = [0, 0]   # automaticaly adjust to first image
    #rescaler = [int(8081/10), int(7981/10)] # automaticaly adjust to first image
    #scale_up = 1000

# -------------------------------------------------------------------------------------------------------
    print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Start', model)
    logFile.write("{} {} {}\n".format(time.strftime("%d.%m.%Y %H:%M:%S"), 'Start', model))
    for band in fileExt:
        with rasterio.open(file_path + displayId + band) as src:
            if (rescaler[0] == 0) | (rescaler[1] == 0):
                rescaler[0] = src.height
                rescaler[1] = src.width
            if ((src.height == rescaler[0]) & (src.width == rescaler[1])):
                data = src.read(1)
                src_data.append(data)
                print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Opening {}{} {}'.format(displayId, band, src_data[len(src_data)-1].shape))
            else:
                # resample data to target shape
                data = src.read(
                    out_shape=(
                        src.count,
                        int(rescaler[0]), # rescale src.height
                        int(rescaler[1]) # src.width
                    ),
                    resampling=Resampling.bilinear
                )
                # scale image transform
                src.ternsform = (src.transform * src.transform.scale(
                    (src.width / data.shape[-1]),
                    (src.height / data.shape[-2])
                ))
                src_data.append(data[0])
                print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Opening {}{} {} *Resized from {}*'.format(displayId, band, src_data[len(src_data)-1].shape,[src.height, src.width]))
# -------------------------------------------------------------------------------------------------------
# save rgb image
    from PIL import Image
    if os.path.isfile(GetFilePath() + '3_CreatingDataset/' + displayId + '.png') == False :
        def norm(band):
            band_min, band_max = band.min(), band.max()
            return ((band - band_min)/(band_max - band_min))

        b2 = norm(src_data[1].astype(np.float)) * 255.0
        b3 = norm(src_data[2].astype(np.float)) * 255.0
        b4 = norm(src_data[3].astype(np.float)) * 255.0
        ba = src_data[1] * src_data[2] * src_data[3]
        ba1 = np.where(np.cumsum(ba, axis=0) == 0, 0, 255)
        ba2 = np.where(np.flip(np.cumsum(np.flip(ba, 0), axis=0), 0) == 0, 0, 255)
        ba = np.where(np.multiply(ba1, ba2) == 0, 0, 255)
        rgba = np.dstack((b4.astype(np.uint8),b3.astype(np.uint8),b2.astype(np.uint8),ba.astype(np.uint8)))
        img_pil = Image.fromarray(rgba)
        img_pil.save(GetFilePath() + '3_CreatingDataset/' + displayId + '.png', "PNG")

        del b2, b3, b4, ba, ba1, ba2
        del img_pil
        del rgba
# -------------------------------------------------------------------------------------------------------
    # prepare model
    print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Loading model')
    logFile.write("{} {}\n".format(time.strftime("%d.%m.%Y %H:%M:%S"), 'Loading model'))
    from GEOUtil_CommonUtils import linear_regr_eq
    dependencies = {'linear_regr_eq': linear_regr_eq}
    # load model
    classifier = tf.keras.models.load_model(model, custom_objects=dependencies)
    # summarize model.
    classifier.summary()

# -------------------------------------------------------------------------------------------------------
    dst_data = np.empty(rescaler[0]*rescaler[1])
    print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Starting data preparation for', os.path.split(model)[-1][:-3])
    logFile.write("{} {}\n".format(time.strftime("%d.%m.%Y %H:%M:%S"), 'Starting data preparation'))
# -------------------------------------------------------------------------------------------------------
    # prepare vector for learning
    Xinput = np.empty([rescaler[0]*rescaler[1],17])
    for band in range(0,12,1):
        print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Building band: B{}'.format(band))
        # metadata
        mtl = toa_utils._load_mtl(file_path + displayId + '_MTL.txt')
        metadata = mtl['L1_METADATA_FILE']
        src_data[band].resize((src_data[band].shape[0]*src_data[band].shape[1]), refcheck=False)
        if (band < 9):
            MR = metadata['RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_{}'.format(band+1)]
            AR = metadata['RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_{}'.format(band+1)]
            SE = metadata['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
            # REFLECTANCE TOA
            Xinput[:,band+6] = src_data[band]
            Xinput[:,band+6] = (MR * Xinput[:,band+6] + AR) / np.sin(np.deg2rad(SE))
        elif band<11:
            K1 = metadata['TIRS_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_{}'.format(band+1)]
            K2 = metadata['TIRS_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_{}'.format(band+1)]
            RM = metadata['RADIOMETRIC_RESCALING']['RADIANCE_MULT_BAND_{}'.format(band+1)]
            RA = metadata['RADIOMETRIC_RESCALING']['RADIANCE_ADD_BAND_{}'.format(band+1)]
            # RADIANCE TOA
            Xinput[:,band+6] = src_data[band]
            Xinput[:,band+6] =  K2 / np.log(K1 / (Xinput[:,band+6]*RM +RA) + 1)
        else:
            # QA Band --> https://landsat.usgs.gov/sites/default/files/documents/landsat_QA_tools_userguide.pdf
            Xinput[:,0] = l8QA.fill_qa(src_data[band])                  # Designated Fill: Areas which are not imaged or could not be populated with data, but are part of the image’s grid
            #Xinput[:,1] = l8QA.terrain_qa(src_data[band])              # Dropped Pixel: An error has occurred during acquisition and erroneous data have been populated into this pixel
            #Xinput[:,2] = l8QA.radiometric_qa(src_data[band])          # Radiometric Saturation: Indicates how many bands contain saturation
            Xinput[:,1] = l8QA.cloud_confidence(src_data[band])         # Cloud Confidence: Confidence in the pixel containing any type of cloud
            Xinput[:,2] = l8QA.cloud(src_data[band])                    # Cloud: Indicates whether or not the pixel contains cloud
            Xinput[:,3] = l8QA.cloud_shadow_confidence(src_data[band])  # Cloud Shadow Confidence: Confidence in the pixel containing cloud shadow
            Xinput[:,4] = l8QA.snow_ice_confidence(src_data[band])      # Snow/Ice Confidence: Confidence in the pixel containing snow and/or ice
            Xinput[:,5] = l8QA.cirrus_confidence(src_data[band])        #  Cirrus Confidence: Confidence in the pixel containing cirrus cloud
    # free memory
    maskImg = np.full([rescaler[0]*rescaler[1]], 1)
    for band in range(0,11,1):
        maskImg[np.where(src_data[band]==0)] = 0
    del src_data
    # scale for predictyion
    print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Scaling')
    #scalarX = MinMaxScaler()
    #X_scaled = scalarX.fit_transform(np.array(Xinput))
    from GEOUtil_CommonUtils import bandsMinMaxScaller
    bands = ['QA_fill',	'QA_cloudConf', 'QA_cloud',	'QA_cloudShadow', 'QA_snowIce', 'QA_cirrusConf', 
         'B1_av', 'B2_av', 'B3_av',	'B4_av', 'B5_av', 'B6_av', 'B7_av', 'B8_av', 'B9_av', 'B10_av', 'B11_av']
    X_scaled = Xinput
    for band in range(0,17,1):
        X_scaled[band] = bandsMinMaxScaller(bands[band], X_scaled[band])
    print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Predicting')
    logFile.write("{} {}\n".format(time.strftime("%d.%m.%Y %H:%M:%S"), 'Predicting'))
    
    # -------------------------------------------------------------------------------------------------------
    # predict results & store
    dst_data = classifier.predict(X_scaled)

    print('dst_data stat   ==0: {}   <0: {}   >0: {}'.format(np.where(dst_data<0)[0].shape[0],
                                                             np.where(dst_data==0)[0].shape[0],
                                                             np.where(dst_data>0)[0].shape[0]))
    logFile.write('dst_data stat   ==0: {}   <0: {}   >0: {}\n'.format(np.where(dst_data<0)[0].shape[0],
                                                             np.where(dst_data==0)[0].shape[0],
                                                             np.where(dst_data>0)[0].shape[0]))
    if dst_data.min() < 0:
        print('WARNING !!!! baseline correction ()'.format(dst_data.min()))
        logFile.write('WARNING !!!! baseline correction ()\n'.format(dst_data.min()))
        dst_data = dst_data - dst_data.min()
    # -------------------------------------------------------------------------------------------------------
    # put the 0 values, when there was 0 in source tables     
    zero_before = np.where(dst_data>0)[0].shape[0]
    dst_data[np.where(maskImg==0)] = 0
    print('Zeros before: {}   after: {}'.format(band, zero_before, np.where(dst_data>0)[0].shape[0]))
    logFile.write('Zeros before: {}   after: {}\n'.format(band, zero_before, np.where(dst_data>0)[0].shape[0]))

    dst_data.resize((rescaler[0],rescaler[1]), refcheck=False)

    #for band in range(0,11,1):
    #    zero_before = np.where(dst_data>0)[0].shape[0]
    #    src_data[band].resize((rescaler[0],rescaler[1]), refcheck=False)
    #    dst_data[np.where(src_data[band]==0)] = 0
    #    print('Zeros for band: {}   before: {}   after: {}'.format(band, zero_before, np.where(dst_data>0)[0].shape[0]))
    negative_before = np.where(dst_data<0)[0].shape[0]
    zero_before = np.where(dst_data==0)[0].shape[0]
    dst_data[(np.where(dst_data<0))] = 0
    dst_type_before = type(dst_data[0,0])
    print('Negative for before: {}   zero_before: {}   after: {}   min: {}   max: {}'.format(negative_before, zero_before, np.where(dst_data==0)[0].shape[0], dst_data.min(), dst_data.max()))
    logFile.write('Negative for before: {}   zero_before: {}   after: {}   min: {}   max: {}\n'.format(negative_before, zero_before, np.where(dst_data==0)[0].shape[0], dst_data.min(), dst_data.max()))
    #dst_data = (dst_data * scale_up).astype(int)
    #print('astype(int) zero: {}   positive: {}   negative: {}   min: {}   max: {}   0xFFFF: {}   types {}-->{}'.format(np.where(dst_data==0)[0].shape[0], 
    #                                                                                     np.where(dst_data>0)[0].shape[0], 
    #                                                                                     np.where(dst_data<0)[0].shape[0],
    #                                                                                     dst_data.min(), dst_data.max(), len(dst_data[dst_data>65535]),
    #                                                                                     dst_type_before, type(dst_data[0,0])))
    #logFile.write('astype(int) zero: {}   positive: {}   negative: {}   min: {}   max: {}   0xFFFF: {}   types {}-->{}\n'.format(np.where(dst_data==0)[0].shape[0], 
    #                                                                                     np.where(dst_data>0)[0].shape[0], 
    #                                                                                     np.where(dst_data<0)[0].shape[0],
    #                                                                                     dst_data.min(), dst_data.max(), len(dst_data[dst_data>65535]),
    #                                                                                     dst_type_before, type(dst_data[0,0])))

# -------------------------------------------------------------------------------------------------------
    #print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Saving np_array {} !!!'.format(dst_file + '.npy'))
    #np.save(dst_file + '.npy', dst_data)

# -------------------------------------------------------------------------------------------------------
    print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Saving TIFF {} !!!'.format(dst_file + '.TIF'))
    logFile.write("{} {}\n".format(time.strftime("%d.%m.%Y %H:%M:%S"), 'Saving'))
    with rasterio.open(file_path + displayId + '_B1.TIF') as src:
        if ((src.height == rescaler[0]) & (src.width == rescaler[1])):
            data = src.read(1)
            print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Opening {}_B{}.TIF {}'.format(displayId, band, data.shape))
        else:
            # resample data to target shape
            data = src.read(
                out_shape=(
                    src.count,
                    int(rescaler[0]), # rescale src.height
                    int(rescaler[1]) # src.width
                ),
                resampling=Resampling.bilinear
            )
            # scale image transform
            src.ternsform = (src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            ))
            data = data[0]
            print(time.strftime("%d.%m.%Y %H:%M:%S"), 'Opening {}_B{}.TIF {} *Resized*'.format(displayId, band, data.shape))
        # https://rasterio.readthedocs.io/en/latest/topics/writing.html
        # Register GDAL format drivers and configuration options with a
        # context manager.
        with rasterio.Env():
            # Write an array as a raster band to a new 8-bit file. For
            # the new file's profile, we start with the profile of the source
            dst_profile = src.profile.copy()
            # And then change the band count to 1, set the
            # dtype to uint16, and specify LZW compression.
            dst_profile.update(
                #driver='GTiff',
                dtype=rasterio.float32,
                #dtype=rasterio.uint16,
                count=1,
                height = int(rescaler[0]),
                width = int(rescaler[1]),
                #crs=src_list[0].crs,
                #compress='lzw'
                )
            with rasterio.open(dst_file + '_v2.TIF', 'w', **dst_profile) as dst:
                dst.write(dst_data.astype(rasterio.float32), 1)
            #    dst.write(dst_data.astype(rasterio.uint16), 1)
            # At the end of the ``with rasterio.Env()`` block, context
            # manager exits and all drivers are de-registered.
    logFile.write("End {} {}\n".format(time.strftime("%d.%m.%Y %H:%M:%S"), 'End'))
    logFile.close()
# -------------------------------------------------------------------------------------------------------
tEnd = datetime.datetime.now().replace(microsecond=0)
print(time.strftime("%d.%m.%Y %H:%M:%S"), '\n\nAll {} done !!!  {}'.format(model, tEnd - tStart))
