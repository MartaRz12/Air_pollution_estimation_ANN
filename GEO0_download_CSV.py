# http://discomap.eea.europa.eu/map/fme/AirQualityExport.htm
# https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw?CountryCode=DE&CityName=&Pollutant=&Year_from=2013&Year_to=2020&Station=&Samplingpoint=&Source=All&Output=HTML&UpdateDate=&TimeCoverage=Year

import os
import sys
import time
import csv
import datetime
import pandas as pd

from GEOUtil_CommonUtils import GetFilePath
file_path = GetFilePath()

stationId = ['STA.DE_DEHE042','STA.DE_DEHE051','STA.DE_DEUB001','STA.DE_DEUB004','STA.DE_DEUB030','STA.DE_DEUB044']

if os.path.isfile(file_path + '0_XLS/STA_DE_timeseries.csv'):
    timeseries = pd.read_csv(file_path + '0_XLS/STA_DE_timeseries.csv', sep = ';')
else:
    timeseries = pd.DataFrame()
print(timeseries)
if os.path.isfile(file_path + '0_XLS/downloaded.csv'):
    fDownloaded = pd.read_csv(file_path + '0_XLS/downloaded.csv', sep = ';')
else:
    fDownloaded = pd.DataFrame(columns = ['File_list'])
print(fDownloaded)
stIdDf = pd.read_csv(file_path + '0_XLS/data on stations.txt')
print (time.strftime("%d.%m.%Y %H:%M:%S"), "Stating")

with open(file_path + '0_XLS/download list.txt', 'r', newline='') as txtInFile:
    line = txtInFile.readline()
    cnt = 1
    while line:
        if (fDownloaded['File_list'] == line[:-2]).any()==True:
            print(cnt, ' Skiping: ', line[:-2])
        else:
            head_tail = os.path.split(line.strip())
            wasError = False
            try:
                tempDf = pd.read_csv(line.strip())
            except:
                print("   -> Error:", sys.exc_info()[0])
                wasError = True
            if wasError == False:
                tempDf = tempDf[['AirQualityStation', 'AirPollutant', 'Concentration', 'DatetimeBegin', 'DatetimeEnd', 'Validity']]
                for id in stationId:
                    tempDfSt = tempDf.loc[(tempDf['Validity'] == 1) & (tempDf['AirQualityStation'] == id)]
                    if (tempDfSt.size > 0):
                        tempDfSt['Latitude'] = (stIdDf.loc[stIdDf['EoICode'] == id[-7:]])['Latitude'].unique()[0]
                        tempDfSt['Longitude'] = (stIdDf.loc[stIdDf['EoICode'] == id[-7:]])['Longitude'].unique()[0]
                        tempDfSt['Altitudee'] = (stIdDf.loc[stIdDf['EoICode'] == id[-7:]])['Altitudee'].unique()[0]
                        timeseries = timeseries.append(tempDfSt)
                fDownloaded = fDownloaded.append({'File_list' : line[:-2]},ignore_index=True)
            if (cnt % 50) == 0:
                if timeseries.size > 0:
                    timeseries.to_csv(file_path + '0_XLS/STA_DE_timeseries.csv', index = False, header=True, sep = ';')
                fDownloaded.to_csv(file_path + '0_XLS/downloaded.csv', index = False, header=True, sep = ';')
                print(time.strftime("%d.%m.%Y %H:%M:%S"), " - Line {}: {}: {}".format(cnt, head_tail[1], timeseries.size), ' * ', wasError)
            else:
                print(time.strftime("%d.%m.%Y %H:%M:%S"), " - Line {}: {}: {}".format(cnt, head_tail[1], timeseries.size), ' ', wasError)
        line = txtInFile.readline()
        cnt += 1
timeseries.to_csv(file_path + '0_XLS/STA_DE_timeseries.csv', index = False, header=True, sep = ';')
fDownloaded.to_csv(file_path + '0_XLS/downloaded.csv', index = False, header=True, sep = ';')
txtInFile.close()
#---------------------------------------------------------------------------------------------------------------------------------------------
format_str = '%Y-%m-%d' # The format
for i,row in timeseries.iterrows():
    if (i % 100) == 0:
        print('At:',time.strftime("%d.%m.%Y %H:%M:%S"), " processing row: ", i)
    timeseries.at[i,'DatetimeBegin'] = datetime.datetime.strptime(row['DatetimeBegin'][0:10], format_str)
    timeseries.at[i,'DatetimeEnd'] = datetime.datetime.strptime(row['DatetimeEnd'][0:10], format_str)
timeseries.to_hdf(file_path + '0_XLS/STA_DE_timeseries.h5', key='df', mode='w')



