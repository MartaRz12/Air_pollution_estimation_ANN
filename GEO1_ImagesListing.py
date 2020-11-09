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

import os
import sys
import time
import datetime

import pandas as pd
import landsatxplore.api

pd.options.mode.chained_assignment = None

from GEOUtil_CommonUtils import GetFilePath
file_path = GetFilePath()
usr = 'Login'
psw = 'Password'

print('Starting ', time.strftime("%d.%m.%Y %H:%M:%S"))
timeseries = pd.read_hdf(file_path + '0_XLS/STA_DE_timeseries.h5', 'df')
print(timeseries)
# -------------------------------------------------------------------------------------------------------
searchQuery = (timeseries[['AirQualityStation', 'Latitude', 'Longitude']]).drop_duplicates()
searchQuery['DataMin'] = ''
searchQuery['DataMax'] = ''
for i,row in searchQuery.iterrows():
    searchQuery.at[i,'DataMax'] = ((timeseries.loc[(timeseries['AirQualityStation'] == row['AirQualityStation']), 'DatetimeBegin']).max())
    searchQuery.at[i,'DataMin'] = ((timeseries.loc[(timeseries['AirQualityStation'] == row['AirQualityStation']), 'DatetimeBegin']).min())
print(searchQuery, '\n\n')
# -------------------------------------------------------------------------------------------------------
pollutantList = (timeseries['AirPollutant']).unique()
print(pollutantList, '\n\n')
# -------------------------------------------------------------------------------------------------------
columsToDrop = ['AirPollutant', 'DatetimeEnd', 'Altitudee', 'Validity','Concentration']
if os.path.isfile(file_path + '1_DataExtract/imageList.csv'):
    imageList = pd.read_csv(file_path + '1_DataExtract/imageList.csv', sep = ';')
else:
    imageList = pd.DataFrame(columns = timeseries.columns.values)
    imageList.drop(columsToDrop, axis=1, inplace=True)
    imageList['entityId'] = ''
    imageList['displayId'] = ''
    imageList['cloudCover'] = ''
    for pollutant in pollutantList:
        imageList[pollutant] = ''
# -------------------------------------------------------------------------------------------------------
searchRec = pd.read_hdf(file_path + '1_DataExtract/searchRec.h5', 'searchRec')
# -------------------------------------------------------------------------------------------------------
format_str = '%Y-%m-%d' # The format

# -------------------------------------------------------------------------------------------------------
def addPollutantToList(il, df, scene):
    #if ((il['DatetimeBegin'] == df['DatetimeBegin'].iloc[0]) & (il['AirQualityStation'] == df['AirQualityStation'].iloc[0])).any():
    #    il.at[(il['DatetimeBegin'] == df['DatetimeBegin'].iloc[0]) & (il['AirQualityStation'] == df['AirQualityStation'].iloc[0]), df['AirPollutant'].iloc[0]] = (df['Concentration'].iloc[0]).mean()
    if (il['displayId'] == scene['displayId']).any():
        #concentr = il.at[(il['displayId'] == scene['displayId']), df['AirPollutant'].iloc[0]]
        il.at[(il['displayId'] == scene['displayId']), df['AirPollutant'].iloc[0]] = (df['Concentration'].iloc[0]).mean()
        return il
    else:
        df['entityId'] = scene['entityId']
        df['displayId'] = scene['displayId']
        df['cloudCover'] = scene['cloudCover']
        df[df['AirPollutant'].iloc[0]] = (df['Concentration']).mean()
        df.drop(columsToDrop, axis=1, inplace=True)
        return il.append(df.iloc[[0]])
# -------------------------------------------------------------------------------------------------------

boolErr = True
errCount = -1
while(boolErr == True):
    errCount += 1
    boolErr = False
    # Initialize a new API instance and get an access key

    for i,row in searchQuery.iterrows():
        if (imageList['AirQualityStation'] == row['AirQualityStation']).any():
            print('Skiping: {}, as it is on the list'.format(row['AirQualityStation']))
        else:
            print('Logging api in: ',   time.strftime("%d.%m.%Y %H:%M:%S"), ' errCount:', errCount)
            api = 0
            try:
                api = landsatxplore.api.API(usr, psw)
                print('   -> api:', api)
            except:
                print('   -> login failed', sys.exc_info()[0])
                boolErr = True
                errCount = errCount + 1
                break
    # -------------------------------------------------------------------------------------------------------
            print('Search:         ',time.strftime("%d.%m.%Y %H:%M:%S"), ' StationId:', row['AirQualityStation'], ' lat:', row['Latitude'], ' lon:', row['Longitude'], ' DataMin:', row['DataMin'], ' DataMax:', row['DataMax'])
            aa = row['DataMin'].strftime("%Y-%m-%d")
            try:
                scenes = api.search(
                    dataset='LANDSAT_8_C1',
                    latitude= float(row['Latitude']),
                    longitude= float(row['Longitude']),
                    start_date = row['DataMin'].strftime("%Y-%m-%d"),
                    end_date = row['DataMax'].strftime("%Y-%m-%d"),
                    max_cloud_cover=100,
                    max_results=1000)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                boolErr = True
                break
            print('End search:     ', time.strftime("%d.%m.%Y %H:%M:%S"), ' ===> {} scenes found.'.format(len(scenes)))
    # -------------------------------------------------------------------------------------------------------
            try:
                api.logout()
            except:
                print("Unexpected error:", sys.exc_info()[0])
            print('Logged api out: ', time.strftime("%d.%m.%Y %H:%M:%S"))
    # -------------------------------------------------------------------------------------------------------
            j = 0
            pollutantList = timeseries.loc[(timeseries['AirQualityStation'] == row['AirQualityStation']),'AirPollutant'].unique()
            for scene in scenes:
                j += 1
                sB = scene['sceneBounds'].split(",")
                if (searchRec['displayId'] == scene['displayId']).any() == False:
                    searchRec = searchRec.append(scene, ignore_index=True)
                    searchRec.to_hdf(file_path + '1_DataExtract/searchRec.h5', key='searchRec', mode='w')
                    searchRec.to_csv(file_path + '1_DataExtract/searchRec.txt', index = False, header=True, sep = ';')
                if (float(sB[0]) < row['Longitude']) & (float(sB[2]) > row['Longitude']) & (float(sB[1]) < row['Latitude']) & (float(sB[3]) > row['Latitude']):
                    tmpDate = scene['acquisitionDate']
                    tmpDateDT = datetime.datetime.strptime(tmpDate, format_str)
                    for pollutant in pollutantList:
                        if ((timeseries['AirQualityStation'] == row['AirQualityStation']) & (timeseries['AirPollutant'] == pollutant) & (timeseries['DatetimeBegin'] == scene['acquisitionDate'])).any():
                            df = timeseries.loc[(timeseries['AirQualityStation'] == row['AirQualityStation']) & (timeseries['AirPollutant'] == pollutant) & (timeseries['DatetimeBegin'] == scene['acquisitionDate'])]
                            imageList = addPollutantToList(imageList, df, scene)
                            print(str(j).rjust(4,' '), ' ', str(imageList.shape).rjust(10,' '), ' Adding: ', scene['acquisitionDate'], ' for ', pollutant.rjust(15,' '), scene['entityId'], scene['displayId'], scene['cloudCover']) 
                        elif ((timeseries['AirQualityStation'] == row['AirQualityStation']) & (timeseries['AirPollutant'] == pollutant)).any():
                            subset = timeseries.loc[((timeseries['AirQualityStation']) == row['AirQualityStation']) & ((timeseries['AirPollutant']) == pollutant)]
                            subset['DataDiff'] = abs(subset['DatetimeBegin'] - tmpDateDT)
                            minval = subset['DataDiff'].min()
                            subset = (subset.loc[subset['DataDiff'] == minval, timeseries.columns.values])
                            if minval.days <= 5:
                                imageList = addPollutantToList(imageList, subset, scene)
                                print(str(j).rjust(4,' '), ' ', str(imageList.shape).rjust(10,' '), ' Using:  ', (subset['DatetimeBegin']).iloc[0].strftime("%Y-%m-%d"), ' for ', pollutant.rjust(15,' '), scene['acquisitionDate'], scene['entityId'], scene['displayId'], scene['cloudCover'])
                            else:
                                print(str(j).rjust(4,' '), '  --> No date:        ', scene['acquisitionDate'], ' for ', pollutant.rjust(15,' '), scene['entityId'], scene['displayId'], scene['cloudCover'])
                        else:
                            print(str(j).rjust(4,' '), '  --> Nothing: ', scene['acquisitionDate'], ' ', pollutant.rjust(15,' '), ' ', scene['entityId'], scene['displayId'], scene['cloudCover'])
                else:
                    print(str(j).rjust(4,' '), '  --> Out of bounds: ', scene['entityId'], scene['displayId'], ' lat: ', row['Latitude'], ' lon: ', row['Longitude'], ' bound: ', scene['sceneBounds'] )
            imageList.to_csv(file_path + '1_DataExtract/imageList.csv', index = False, header=True, sep = ';')