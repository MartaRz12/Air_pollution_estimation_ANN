# -------------------------------------------------------------------------------------------------------
# description of satelite images: 
#   https://gisgeography.com/landsat-8-bands-combinations/
#   https://landsat.gsfc.nasa.gov/landsat-8/landsat-8-bands/
#   https://www.tandfonline.com/doi/abs/10.1080/10106049.2015.1094522
#   https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-oli-operational-land-imager-and?qt-science_center_objects=0#qt-science_center_objects
# Images downloaded from:
#   https://earthexplorer.usgs.gov
# free acount needed
# Login: Login
# Password: Password

# Information about Landsat8 bands with CO2 corellation:
# https://www.tandfonline.com/doi/figure/10.1080/10106049.2015.1094522?scroll=top&needAccess=true
# -------------------------------------------------------------------------------------------------------

import pandas as pd
import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import time
from GEOUtil_UnTar_gz import file_untar
import os
import sys
import requests
from bs4 import BeautifulSoup
import shutil
import csv
import threading

from GEOUtil_CommonUtils import GetFilePath
file_path = GetFilePath()
image_path = GetFilePath() + 'temp/'
usr = 'Login'
psw = 'Password'

# -------------------------------------------------------------------------------------------------------
#s3_scenes = pd.read_csv('http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz', compression='gzip')
#google_scenes = pd.read_csv('https://storage.googleapis.com/gcp-public-data-landsat/index.csv.gz', compression='gzip')
print('Load AmazonDB:  ', time.strftime("%d.%m.%Y %H:%M:%S"))
s3_scenes_a = pd.read_csv(file_path + 'scene_list_amazon.csv.zip', compression='zip')
print('AmazonDB loaded:', time.strftime("%d.%m.%Y %H:%M:%S"))
# -------------------------------------------------------------------------------------------------------
df = pd.read_csv(file_path + '1_DataExtract/imageList.csv', sep = ';')
print(df)
# -------------------------------------------------------------------------------------------------------
badFiles = pd.read_csv(file_path + '1_DataExtract/badFiles.txt', sep = ';')
print(badFiles)

# -------------------------------------------------------------------------------------------------------
def chkFileExist(image_id):
    fileExt = ['_B1.TIF', '_B2.TIF', '_B3.TIF', '_B4.TIF', '_B5.TIF', '_B6.TIF', '_B7.TIF', '_B8.TIF', '_B9.TIF', '_B10.TIF', '_B11.TIF', '_BQA.TIF', '_MTL.txt', '_ANG.txt']
    numFi = 0
    for fext in fileExt:
        if os.path.isfile(image_path + image_id + fext):
            numFi += 1
    if numFi != len(fileExt):
        return False
    return True

# -------------------------------------------------------------------------------------------------------
# http://geologyandpython.com/get-landsat-8.html
# -------------------------------------------------------------------------------------------------------
def downloadAmazon(image_id):
    print('Amazon started: ', time.strftime("%d.%m.%Y %H:%M:%S"), ' for:', image_id)
    scens = s3_scenes_a[s3_scenes_a.productId == image_id]
    if len(scens):
        url = scens.iloc[0].download_url
        print('List request:   ', time.strftime("%d.%m.%Y %H:%M:%S"))
        try:
            response = requests.get(url)
        except:
            print('Responses Err List:', time.strftime("%d.%m.%Y %H:%M:%S"))
            return -5
        if response.status_code == 200:
            print('Files list:     ', time.strftime("%d.%m.%Y %H:%M:%S"))
            html = BeautifulSoup(response.content, 'html.parser')
            for li in html.find_all('li'):
                file = li.find_next('a').get('href')
                if ((file[-3:] == 'TIF') | (file[-7:] == 'MTL.txt') | (file[-7:] == 'ANG.txt')) & (os.path.isfile(image_path + file) == False) & (os.path.isfile(image_path + 'tempA/' + file) == False):
                    print('  Downloading: {} '.format(file), time.strftime("%d.%m.%Y %H:%M:%S"))
                    try:
                        response = requests.get(url.replace('index.html', file), stream=True)
                    except:
                        print('Responses Err download:', time.strftime("%d.%m.%Y %H:%M:%S"))
                        return -4
                    with open(os.path.join(image_path + 'tempA/', file), 'wb') as output:
                        shutil.copyfileobj(response.raw, output)
                    del response
                else:
                    print('  Skiping: {}     '.format(file), time.strftime("%d.%m.%Y %H:%M:%S"))
            print('Move Img        ', time.strftime("%d.%m.%Y %H:%M:%S"))
            filelist = [ f for f in os.listdir(image_path + 'tempA/')]
            for f in filelist:
                os.replace(image_path + 'tempA/' + f, image_path + f)
        else:
            print('No responses:   ', time.strftime("%d.%m.%Y %H:%M:%S"))
            return -2
    else:
        print('No files:       ', time.strftime("%d.%m.%Y %H:%M:%S"))
        return -1
    print('Finnished at:   ', time.strftime("%d.%m.%Y %H:%M:%S"))
    return 1

# -------------------------------------------------------------------------------------------------------
# https://pypi.org/project/landsatxplore/
# -------------------------------------------------------------------------------------------------------
def downloadEartExplorer(displayId, entityId, badFiles):
    if chkFileExist(displayId):
        print('  All files: {}     '.format(displayId), time.strftime("%d.%m.%Y %H:%M:%S"))
        return 1
    print('Logging ee in:  ',   time.strftime("%d.%m.%Y %H:%M:%S"))
    ee = EarthExplorer(usr, psw)
    print('   -> ee: ',ee)
    print('Download        ', time.strftime("%d.%m.%Y %H:%M:%S"), displayId, entityId)
    try:
        ee.download(scene_id=entityId, output_dir=image_path + 'tempE/')
    except IOError as e:
        print("   -> I/O error({0}): {1}".format(e.errno, e.strerror))
        return -1
    except:
        print("   -> Unexpected error:", sys.exc_info()[0])
        return -2
    print('UnTarGz         ', time.strftime("%d.%m.%Y %H:%M:%S"))
    if (file_untar(image_path + 'tempE/', displayId + '.tar.gz') < 0):
        print('   -> unTar error: ', image_path + 'tempE/', image_id + '.tar.gz')
        return -3
    print('Move Img        ', time.strftime("%d.%m.%Y %H:%M:%S"))
    filelist = [ f for f in os.listdir(image_path + 'tempE/')]
    for f in filelist:
        os.replace(image_path + 'tempE/' + f, image_path + f)
    #if os.path.isfile(image_path + image_id + '_B10.TIF') == False or os.path.isfile(image_path + image_id + '_B11.TIF') == False:
    #    print('   -> no file in directory error: ', image_id, ' -- ', os.path.isfile(image_path + image_id + '_B10.TIF'), ' -- ', os.path.isfile(image_path + image_id + '_B11.TIF'))
    #    if (badFiles['badFile'] == image_id).any():
    #        badFiles = badFiles.append({'badFile' : image_id}, ignore_index=True)
    #        print('   -> Bad file was added to the list !!!!!!!!!:', image_id)
    #        badFiles.to_csv(file_path + '1_DataExtract/badFiles.txt', index = False, header=True, sep = ';')
    #    return -4
    print('Loging ee out:  ', time.strftime("%d.%m.%Y %H:%M:%S"))
    ee.logout()
    print('Finnished at:   ', time.strftime("%d.%m.%Y %H:%M:%S"))
    return 1

# -------------------------------------------------------------------------------------------------------
def AmazonLoop():
    j = 0
    for row in df['displayId']:
        print('Clean temp      ', time.strftime("%d.%m.%Y %H:%M:%S"))
        filelist = [ f for f in os.listdir(image_path + 'tempA/')]
        for f in filelist:
            os.remove(os.path.join(image_path + 'tempA/', f))
        j += 1
        print('Downoading ', row, ' ', time.strftime("%d.%m.%Y %H:%M:%S"), "(", j, "/", df.shape[0], ")")
        if chkFileExist(row) == False:
            downloadAmazon(row)
        else:
            print('  All files: {}     '.format(row), time.strftime("%d.%m.%Y %H:%M:%S"))

# -------------------------------------------------------------------------------------------------------
def EELoop():
    for i,row in df.iterrows():
        print('Clean temp      ', time.strftime("%d.%m.%Y %H:%M:%S"))
        filelist = [ f for f in os.listdir(image_path + 'tempE/')]
        for f in filelist:
            os.remove(os.path.join(image_path + 'tempE/', f))
        print('Downoading ', row['displayId'], ' ', time.strftime("%d.%m.%Y %H:%M:%S"), "(", i, "/", df.shape[0], ")")
        if downloadEartExplorer(row['displayId'], row['entityId'], badFiles) < 0:
            print('Downloads error:', time.strftime("%d.%m.%Y %H:%M:%S"))
            break


#downloadAmazon('LC08_L1TP_188025_20190301_20190309_01_T1')     # img of polish city -> Katowice
#downloadAmazon('LC08_L1TP_188025_20190317_20190325_01_T1')     # img of polish city -> Katowice
#downloadAmazon('LC08_L1TP_189025_20190324_20190403_01_T1')     # img of polish city -> Katowice
#downloadAmazon('LC08_L1TP_189023_20190324_20190403_01_T1')     # img of polish city -> Płock
#downloadAmazon('LC08_L1TP_188023_20190301_20190309_01_T1')     # img of polish city -> Płock
#downloadAmazon('LC08_L1TP_190022_20190331_20190404_01_T1')     # img of polish city -> Gdańsk
#downloadAmazon('LC08_L1TP_191022_20190306_20190309_01_T1')     # img of polish city -> Gdańsk


doMultiThread = False
if doMultiThread:
    threads = []
    t = threading.Thread(target=AmazonLoop)
    threads.append(t)
    t.start()
    t = threading.Thread(target=EELoop)
    threads.append(t)
    t.start()
else:
    AmazonLoop()
    EELoop()
