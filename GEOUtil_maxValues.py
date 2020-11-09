import os
from glob import glob # File manipulation
import numpy as np
import pandas as pd
import rasterio
import time
import matplotlib.pyplot as plt
from rio_toa import toa_utils
from rio_toa import sun_utils

# -------------------------------------------------------------------------------------------------------
from GEOUtil_CommonUtils import GetFilePath
from GEOUtil_CommonUtils import GetImagesPath
file_path = GetFilePath()
image_path = GetImagesPath()

doHist = False
iloscP = 10
skok = 5.0
os.chdir(file_path)
os.listdir(file_path)
fileExt = ['_B1.TIF', '_B2.TIF', '_B3.TIF', '_B4.TIF', '_B5.TIF', '_B6.TIF', '_B7.TIF', '_B8.TIF', '_B9.TIF', '_B10.TIF', '_B11.TIF']
#fileExt = ['_B2.TIF']

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------
df = pd.read_csv(file_path + '1_DataExtract/imageList.csv', sep = ';')
searchRec = pd.read_hdf(file_path + '1_DataExtract/searchRec.h5', 'searchRec')
print('{}\n'.format(df))

ccol = []
cval = []
for fe in fileExt:
    ccol.append('max' + fe[:-4])
    ccol.append('min' + fe[:-4])
    ccol.append('mx' + fe[:-4])
    ccol.append('mn' + fe[:-4])
    cval.append(-1000000)  
    cval.append(1000000)
    cval.append(-1000000.0)
    cval.append(1000000.0)

dfTemp = pd.DataFrame([cval,cval], columns=ccol)
pd.reset_option('mode.chained_assignment')

totC = []
step = 50
bin_range = np.arange(0, 65535+step, step)

for i,row in df.iterrows():
    image_id = row['displayId']
    if os.path.isfile(image_path + image_id + '_MTL.txt'):
        mtl = toa_utils._load_mtl(image_path + image_id + '_MTL.txt')
        metadata = mtl['L1_METADATA_FILE']
    
        for fe in fileExt:
            srcFile = image_path + image_id + fe
            band = int(fe[2:-4])
            if os.path.isfile(srcFile):
                with  rasterio.open(srcFile) as bandImg:
                    band_in = bandImg.read(1)
                    band_in.resize((band_in.shape[0]*band_in.shape[1]), refcheck=False)
                    for scope in range(0, 2, 1):
                        if scope == 1:
                            band_in = band_in[np.where(band_in>0)]
                            band_in = band_in[np.where(band_in<0xFFFF)]
                        max = band_in.max()
                        min = band_in.min()
                        if max > dfTemp['max' + fe[:-4]][scope]:
                            with pd.option_context('mode.chained_assignment', None):
                                dfTemp['max' + fe[:-4]][scope] = max
                        if min < dfTemp['min' + fe[:-4]][scope]:
                            with pd.option_context('mode.chained_assignment', None):
                                dfTemp['min' + fe[:-4]][scope] = min
                        #----------------------------------------------------------------------------------------------
                        if (scope==0 & doHist):
                            count, division = np.histogram(band_in, bins=bin_range)
                            if len(totC) == 0:
                                totC = count
                            else:
                                totC = totC + count
                        #----------------------------------------------------------------------------------------------
                        #count, division  = pd.cut(s, bins=bin_range, include_lowest=True, right=False, retbins=True)

                        #----------------------------------------------------------------------------------------------
                        if (band < 10):
                            MR = metadata['RADIOMETRIC_RESCALING']['REFLECTANCE_MULT_BAND_{}'.format(band)]
                            AR = metadata['RADIOMETRIC_RESCALING']['REFLECTANCE_ADD_BAND_{}'.format(band)]
                            SE = metadata['IMAGE_ATTRIBUTES']['SUN_ELEVATION']
                            # REFLECTANCE TOA
                            Xinput = (MR * band_in + AR) / np.sin(np.deg2rad(SE))
                        else: # band 10, 11
                            K1 = metadata['TIRS_THERMAL_CONSTANTS']['K1_CONSTANT_BAND_{}'.format(band)]
                            K2 = metadata['TIRS_THERMAL_CONSTANTS']['K2_CONSTANT_BAND_{}'.format(band)]
                            RM = metadata['RADIOMETRIC_RESCALING']['RADIANCE_MULT_BAND_{}'.format(band)]
                            RA = metadata['RADIOMETRIC_RESCALING']['RADIANCE_ADD_BAND_{}'.format(band)]
                            # RADIANCE TOA
                            Xinput =  K2 / np.log(K1 / (band_in*RM +RA) + 1)

                        max = Xinput.max()
                        min = Xinput.min()

                        if max > dfTemp['mx' + fe[:-4]][scope]:
                            with pd.option_context('mode.chained_assignment', None):
                                dfTemp['mx' + fe[:-4]][scope] = max
                        if min < dfTemp['mn' + fe[:-4]][scope]:
                            with pd.option_context('mode.chained_assignment', None):
                                dfTemp['mn' + fe[:-4]][scope] = min
                        #----------------------------------------------------------------------------------------------
                        print(str(i).rjust(5,' '), '/', str(df.shape[0]).rjust(5,' '), time.strftime("  -> %d.%m.%Y %H:%M:%S"), image_id + fe, 'scope:', scope,
                              'min:', dfTemp['min' + fe[:-4]][scope], 'max:', dfTemp['max' + fe[:-4]][scope],
                              'mx:', dfTemp['mx' + fe[:-4]][scope], 'mn:', dfTemp['mn' + fe[:-4]][scope])

        if (doHist & i>10): break

if doHist:
    import matplotlib.pyplot as plt

    #plt.bar(range(len(totC[1:])), totC[1:], color = 'b', width = 0.25)
    plt.plot(bin_range[1:-1], totC[1:], 'ro')
    plt.show()
    print(dfTemp)

for scope in range(0, 2, 1):
    for fe in fileExt:
        print('scope', scope,
              'min' + fe, dfTemp['min' + fe[:-4]][scope], 'max' + fe, dfTemp['max' + fe[:-4]][scope],
              'mn' + fe, dfTemp['mn' + fe[:-4]][scope], 'mx' + fe, dfTemp['mx' + fe[:-4]][scope])
dfTemp.to_csv(file_path + '1_DataExtract/min_max_stat.csv', index = False, header=True, sep = ';')
