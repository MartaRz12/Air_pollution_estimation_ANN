
# -------------------------------------------------------------------------------------------------------
# Measuring the accurancy for regresion model
# https://stackoverflow.com/questions/50797135/why-does-this-neural-network-have-zero-accuracy-and-very-low-loss
# -------------------------------------------------------------------------------------------------------
import sys

try:
    import keras.backend as K
    import tensorflow as tf

    accepted_diff = 0.01

    def linear_regr_eq(y_true, y_pred):
        diff = K.abs(y_true-y_pred)
        return K.mean(K.cast(diff < accepted_diff, tf.float32))

    def getAcceptedDiff():
        return accepted_diff
except:
    print("WARNING! GeoUtil - CommonUtils - error Kerras import", sys.exc_info()[0])

# -------------------------------------------------------------------------------------------------------
# file paths
# -------------------------------------------------------------------------------------------------------
my_file_path = r'./GeoProject/'
def GetFilePath():
    return my_file_path

#my_image_path = r'./2_Images/'
my_image_path = r'./GeoProject/temp/'
def GetImagesPath():
    return my_image_path


# -------------------------------------------------------------------------------------------------------
# MinMax scaller
# -------------------------------------------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    bandStats = pd.read_csv(my_file_path + '1_DataExtract/min_max_stat.csv', sep = ';')

    dfPoluntanMinMax = pd.DataFrame({'pollutant':['Pb in PM10','Indeno-(1;2;3-cd)pyrene in PM','Na+  in PM2.5','NO3- in PM2.5','Benzo(k)fluoranthene in PM10','SO2','CH2=CH-CH=CH2','C6H14','NH4+ in PM2.5','SO42- in PM2.5','Co in PM10','Benzo(a)anthracene','H3C-CH(CH3)2','BS','H2C=CH-CH2-CH3','H3C-CH2-CH(CH3)2','NH3','V in PM10','NOX as NO2','O3','Cu in PM10','Ni in PM10','As in PM10','NO','CH4','Ca2+ in PM2.5','Mn in PM10','Cd in PM10','PM10','Pb','(CH3)2-CH-CH2-CH2-CH3','C6H5-C2H5','OC in PM2.5','Dibenzo(ah)anthracene in PM10','cis-H3C-CH=CH-CH3','C7H16','BaP in PM10','Mg2+ in PM2.5','C2H6','CO2','NO2','As','Dibenzo(ah)anthracene','Cd','Hg0 + Hg-reactive','Ni','Cl- in PM2.5','HC=CH','Benzo(b;j)fluorantheneinPM10','K+ in PM2.5','PM2.5','Hg','Benzo(a)anthracene in PM10','Benzo(b;j;k)fluoranthene','Indeno-(1;2;3-cd)pyrene','H2C=CH2','H3C-(CH2)3-CH3','EC in PM2.5','CO','o-C6H4-(CH3)2','C6H5-CH3','C6H6','CH2=CH-CH3','THC (NM)','CH2=CH-C(CH3)=CH2','trans-H3C-CH=CH-CH3','H3C-CH2-CH3','m;p-C6H4(CH3)2','H3C-CH2-CH2-CH3','BaP','N2O','Benzo(b;j;k)fluorantheneInPM1','Pentenes'],
                                     'MaxVal'   :[0.022,0.855,2.082,34.794,0.523,78.006,0.441,1.886,11.347,12.245,0.23,23.043,1.924,4.27,6.543,13.543,30.579,2.36,120.704,159.2,160.73,7.3,7.284,73.975,1528.4,0.48,38,0.64,1820.18,8614.67,2.286,0.627,13.048,0.087,0.264,1.087,0.578,0.256,8.445,966.913,66.07,2378.49,7.45,388.52,2.237,9270.54,3.523,3.21,1.587,0.446,70.099,202.13,0.39,128.301,51.936,6.232,1.884,1.859,0.945,0.869,4.183,4.368,3.364,31.898,3.724,0.413,8.013,1.364,3.173,25.899,475.515,0.913,0.592],
                                     'MinVal'   :[0.0,0.003,0.009,0.026,0.006,0.05,0.004,0.003,0.024,0.026,0.006,0,0.048,0.01,0.02,0.029,0.001,0.026,0.25,0.6,0.397,0.042,0.01,-0.07,1.211,0.001,0.099,0.003,0.26,-777,0.007,0.008,0.132,0,0.006,0.004,0.001,0.001,0.617,379,0.1,-777,0,-777,0.966,-777,0.008,0.061,0.008,0.005,0.712,-777,0.001,0,0,0.03,0.011,0,0.067,0.013,0.049,0.012,0.124,3.278,0.002,0.039,0.054,0.017,0.048,0,327.2,0.01,0.049]}) 

except:
    print("WARNING! GeoUtil - CommonUtils - error pandas import", sys.exc_info()[0])


def bandsMinMaxScaller(band, xCol):
    #mn, ptp = x.min(), x.ptp()
    #x_scaled = (x - mn) / ptp

    # mn, mx = x.min(), x.max()
    # x_scaled = (x - mn) / (mx - mn)

    if (dfPoluntanMinMax['pollutant'] == band).any(): # to scale the Y based on the pollutant dictionary
        mn = dfPoluntanMinMax[dfPoluntanMinMax['pollutant'] == band]['MinVal'].item()
        mx = dfPoluntanMinMax[dfPoluntanMinMax['pollutant'] == band]['MaxVal'].item()
        y_scaled = (xCol - mn) / (mx - mn)
        return y_scaled
    elif band == 'QA_fill': # to scale the X QA band 
        return xCol
    elif band == 'QA_cloudConf':
        return xCol/3.0
    elif band == 'QA_cloud':
        return xCol
    elif band == 'QA_cloudShadow':
        return xCol/3.0
    elif band == 'QA_snowIce':
        return xCol/3.0
    elif band == 'QA_cirrusConf':
        return xCol/3.0
    else: # to scale the X band based on statisicis from file
        scaleType = 1 # type 0 - the scale includes 0 & 0xFFFF; type 1 - the scale excluded 0 & 0xFFFF, as 0 is error and 0xFFFF is saturation
        mn = bandStats['mn_'+band[:-3]][scaleType]
        mx = bandStats['mx_'+band[:-3]][scaleType]

        x_scaled = (xCol - mn) / (mx - mn)
        if type(x_scaled).__name__ == 'ndarray':
            x_scaled[np.where(x_scaled > 1.0)] = 1.0
            x_scaled[np.where(x_scaled < 0.0)] = 0.0
        else:
            x_scaled.loc[x_scaled > 1.0] = 1.0
            x_scaled.loc[x_scaled < 0.0] = 0.0
        return x_scaled

