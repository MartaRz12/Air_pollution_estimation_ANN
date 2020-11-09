# https://stackoverflow.com/questions/19043846/how-to-correctly-project-a-tif-image-using-matplotlib-basemap
# https://gadm.org/download_country_v3.html
# https://www.geophysique.be/2010/05/05/matplotlib-basemap-tutorial-part-01-your-first-map/

import os                                   # conda install os
import sys
import numpy as np                          # conda install numpy
import time                                 # conda install time
import matplotlib.pylab as plt              # conda install matplotlib
import gdal                                 # conda install gdal
from rio_toa import toa_utils               # pip install rio-toa
import math                                 # conda install math
import mpl_toolkits                         # conda install mpl_toolkits
from mpl_toolkits.basemap import Basemap    # conda install -c conda-forge basemap-data-hires=1.0.8.dev0
# conda udate --all

from GEOUtil_CommonUtils import GetFilePath
from GEOUtil_CommonUtils import GetImagesPath
file_path = GetFilePath() + '2_Images/'
#my_file = GetFilePath() + '3_CreatingDataset/LC08_L1TP_196025_20190410_20190422_01_T1_v2_SO2_mae_v2.TIF'
#my_file = GetFilePath() + '3_CreatingDataset/LC08_L1TP_196025_20190410_20190422_01_T1_v2_NOX as NO2_mae_v2.TIF'
#my_file = GetFilePath() + '3_CreatingDataset/LC08_L1TP_196025_20190410_20190422_01_T1_v2_O3_mae_v2.TIF'
#my_file = GetFilePath() + '3_CreatingDataset/LC08_L1TP_196025_20190410_20190422_01_T1_v2_O3_mae_v2.TIF'
#my_file = GetFilePath() + '3_CreatingDataset/LC08_L1TP_196025_20190410_20190422_01_T1_v2_PM10_mae_v2.TIF'
#my_file = GetFilePath() + '3_CreatingDataset/LC08_L1TP_196025_20190410_20190422_01_T1_v2_PM10_mae_v2.TIF'
my_file = GetFilePath() + '3_CreatingDataset/LC08_L1TP_190022_20190331_20190404_01_T1_v2_PM10_mae_v2.TIF'
#my_file = GetFilePath() + '3_CreatingDataset/LC08_L1TP_190022_20190331_20190404_01_T1_v2_SO2_mae_v2.TIF'

fname = (os.path.split(my_file)[-1]).split('_')
my_txt_file = GetImagesPath() + fname[0] + '_' + fname[1] + '_' + fname[2] + '_' + fname[3] + '_' + fname[4] + '_' + fname[5] + '_' + fname[6] + '_MTL.txt'
imgRGBAFile = GetFilePath() + '3_CreatingDataset/' + fname[0] + '_' + fname[1] + '_' + fname[2] + '_' + fname[3] + '_' + fname[4] + '_' + fname[5] + '_' + fname[6] + '.png'

lonLat_scaler = 5
map_margin = 0.5

# load image
raster = gdal.Open(my_file, gdal.GA_ReadOnly)
array = raster.GetRasterBand(1).ReadAsArray()
msk_array = np.ma.masked_where(((array ==0) | (array ==65535)), array)
#msk_array = np.ma.masked_equal(array, value = 65535)
print ('Raster Projection:\n', raster.GetProjection())
print ('Raster GeoTransform:\n', raster.GetGeoTransform())

mlt = None
# find file & load lon lat cooridanes from image
for fdir in os.walk(GetImagesPath()):
    my_txt_file = os.path.join(fdir[0], fname[0] + '_' + fname[1] + '_' + fname[2] + '_' + fname[3] + '_' + fname[4] + '_' + fname[5] + '_' + fname[6] + '_MTL.txt')
    if os.path.isfile(my_txt_file):
        mtl = toa_utils._load_mtl(my_txt_file)
        break
metadata = mtl['L1_METADATA_FILE']
CORNER_UL_LON_PRODUCT = metadata['PRODUCT_METADATA']['CORNER_UL_LON_PRODUCT']
CORNER_LL_LAT_PRODUCT = metadata['PRODUCT_METADATA']['CORNER_LL_LAT_PRODUCT']
CORNER_UR_LON_PRODUCT = metadata['PRODUCT_METADATA']['CORNER_UR_LON_PRODUCT']
CORNER_UR_LAT_PRODUCT = metadata['PRODUCT_METADATA']['CORNER_UR_LAT_PRODUCT']
# map coordinates
ul_lon = math.floor(CORNER_UL_LON_PRODUCT) - map_margin
ll_lat = math.floor(CORNER_LL_LAT_PRODUCT) - map_margin
ur_lon = math.ceil(CORNER_UR_LON_PRODUCT) + map_margin
ur_lat = math.ceil(CORNER_UR_LAT_PRODUCT) + map_margin
print('Map coordinates : {},{},{},{}'.format(ul_lon, ll_lat, ur_lon, ur_lat))

#ul_lon = 0.0
#ll_lat = 40.0
#ur_lon = 20.0
#ur_lat = 60.0

# prepare mapplot
fig = plt.figure(figsize=(11.7,8.3))

#Custom adjust of the subplots
plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,wspace=0.15,hspace=0.05)
ax = plt.subplot(111)
# https://matplotlib.org/basemap/api/basemap_api.html
# https://basemaptutorial.readthedocs.io/en/latest/backgrounds.html#warpimage
# https://www.datadependence.com/2016/06/creating-map-visualisations-in-python/
# https://nordicesmhub.github.io/deep_python/14-publish/index.html
# resolution:
# The ‘resolution’ argument is the quality of the map you are creating. 
# The options are crude, low, intermediate, high or full. 
# The higher the resolution the longer it takes to render the map, and it can take a very long time so I recommend that while you are working on your map, 
# set it to crude, and then if and when you want to publish it set it to full.
map = Basemap(resolution='f', projection='merc', llcrnrlat=ll_lat,urcrnrlat=ur_lat,llcrnrlon=ul_lon,urcrnrlon=ur_lon,lat_ts=51.0)
#map.shadedrelief()
try:
    map.etopo()
except:
    print("--> drawcountries error: ", sys.exc_info()[0])
try:
    map.drawcountries(linewidth=0.5)
except:
    print("--> drawcountries error: ", sys.exc_info()[0])
try:
    map.drawcoastlines(linewidth=0.5)
except:
    print("--> drawcoastlines error: ", sys.exc_info()[0])
try:
    map.drawrivers(color='#0000ff')
except:
    print("--> drawrivers error: ", sys.exc_info()[0])

map.drawparallels(np.arange(ll_lat,ur_lat,(ur_lat-ll_lat)/10),labels=[1,0,0,0],color='black',dashes=[1,0],labelstyle='+/-',linewidth=0.2) # draw parallels
map.drawmeridians(np.arange(ul_lon,ur_lon,(ur_lon-ul_lon)/10),labels=[0,0,0,1],color='black',dashes=[1,0],labelstyle='+/-',linewidth=0.2) # draw meridians

datain = np.flipud( msk_array )

nx = raster.RasterXSize
ny = raster.RasterYSize

# position image to map in pixels
dxmin = (map.xmax - map.xmin)*(CORNER_UL_LON_PRODUCT-ul_lon)/(ur_lon-ul_lon)
dxmax = (map.xmax - map.xmin)*(CORNER_UR_LON_PRODUCT-ul_lon)/(ur_lon-ul_lon)
dymin = (map.ymax - map.ymin)*(CORNER_LL_LAT_PRODUCT-ll_lat)/(ur_lat-ll_lat)
dymax = (map.ymax - map.ymin)*(CORNER_UR_LAT_PRODUCT-ll_lat)/(ur_lat-ll_lat)
xin = np.linspace(map.xmin+dxmin,map.xmin+dxmax,nx) # nx is the number of x points on the grid
yin = np.linspace(map.ymin+dymin,map.ymin+dymax,ny) # ny in the number of y points on the grid

lons = np.linspace(CORNER_UL_LON_PRODUCT,CORNER_UR_LON_PRODUCT, int(nx/lonLat_scaler)) #from raster.GetGeoTransform()
lats  = np.linspace(CORNER_LL_LAT_PRODUCT,CORNER_UR_LAT_PRODUCT, int(ny/lonLat_scaler)) 

lons, lats = np.meshgrid(lons,lats) 
xout,yout = map(lons, lats)
dataout = mpl_toolkits.basemap.interp(datain, xin, yin, xout, yout, order=1)

# display the scale metrix
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
levels = np.linspace(dataout.min(),dataout.max(),11)
# https://scipy-cookbook.readthedocs.io/items/Matplotlib_Show_colormaps.html
# reversed color map can be obtained, buy assing "_r" to the color name 
#cntr = map.contourf(xout,yout,dataout, levels, cmap=plt.cm.Reds, mincnt=1, transparent=True)
#cntr = map.contourf(xout,yout,dataout, levels, cmap=plt.cm.jet, mincnt=1, transparent=True)
#cntr = map.contourf(xout,yout,dataout, levels, cmap=plt.cm.YlOrRd, mincnt=1, transparent=True)
cntr = map.contourf(xout,yout,dataout, levels, cmap=plt.cm.RdYlGn_r, mincnt=1, transparent=True)
cbar = map.colorbar(cntr,location='bottom',pad='15%')
#cbar = map.colorbar(cntr,location='right',pad='15%')

# display city names
# https://peak5390.wordpress.com/2012/12/08/matplotlib-basemap-tutorial-plotting-points-on-a-simple-map/
cities = [['Düsseldorf',        51.2277, 6.7735],
          ['Stuttgart',         49.1427, 9.2109],
          ['Frankfurt',         50.1109, 8.6821],
          ['Ludwigshafen',      49.4875, 8.4660],
          ['Kaiserslautern',    49.4401, 7.7491],
          ['Kolonia',           50.9375, 6.9603],
          ['Ouren',             50.1389, 6.1326],
          ['Siegburg',          50.7998, 7.2075],
          ['Warszawa',          52.2297, 21.0122],
          ['Lublin',            51.2465, 22.5684],
          ['Kozienice',         51.5855, 21.5512],
          ['Katowice',          50.2649, 19.0238],
          ['Siedlce',           52.1676, 22.2902],
          ['Płock',             52.5463, 19.7065],
          ['Gdańsk',            54.3520, 18.6466],
          ['Bełchatów',         51.3688, 19.3564],
          ['Poznań',            52.4064, 16.9252],
          ['Łódź',              51.7592, 19.4560],
          ['Kraków',            50.0647, 19.9450],
          ['Zakpoane',          49.2992, 19.9496],
          ['IMG LL',            metadata['PRODUCT_METADATA']['CORNER_LL_LAT_PRODUCT'], metadata['PRODUCT_METADATA']['CORNER_LL_LON_PRODUCT']],
          ['IMG UR',            metadata['PRODUCT_METADATA']['CORNER_UR_LAT_PRODUCT'], metadata['PRODUCT_METADATA']['CORNER_UR_LON_PRODUCT']],
          ['IMG LR',            metadata['PRODUCT_METADATA']['CORNER_LR_LAT_PRODUCT'], metadata['PRODUCT_METADATA']['CORNER_LR_LON_PRODUCT']],
          ['IMG UL',            metadata['PRODUCT_METADATA']['CORNER_UL_LAT_PRODUCT'], metadata['PRODUCT_METADATA']['CORNER_UL_LON_PRODUCT']]
         ]
cities = np.array(cities)
loc_name  = cities[:,0]
lats = cities[:,1].astype(float)
lons = cities[:,2].astype(float)
x,y = map(lons, lats)

for i in range(0, len(loc_name),1):
    map.plot(x[i], y[i], 'bo', markersize=5)
    x2, y2 = map(lons[i], lats[i])
    plt.annotate(loc_name[i], xy=(x[i], y[i]),  xycoords='data',
                    xytext=(x2+100, y2+8000), textcoords='data',
                    arrowprops=dict(arrowstyle="->"))
plt.title(fname[8] + ' --- ' + fname[4][0:4] + '/' + fname[4][4:6] + '/' + fname[4][6:8])


def printFullPicOnMap(pic):
    x0, y0 = map(CORNER_UL_LON_PRODUCT, CORNER_LL_LAT_PRODUCT)
    x1, y1 = map(CORNER_UR_LON_PRODUCT, CORNER_UR_LAT_PRODUCT)
    #x0, y0 = map(metadata['PRODUCT_METADATA']['CORNER_UL_LON_PRODUCT'], metadata['PRODUCT_METADATA']['CORNER_LR_LAT_PRODUCT'])
    #x1, y1 = map(metadata['PRODUCT_METADATA']['CORNER_UR_LON_PRODUCT'], metadata['PRODUCT_METADATA']['CORNER_UR_LAT_PRODUCT'])
    plt.imshow(plt.imread(pic), origin='upper', zorder=1, alpha=1., extent= (x0, x1, y0, y1))
#printFullPicOnMap(imgRGBAFile)

def printThumbPic(pic):
    axicon = fig.add_axes([0.06, 0.08, 0.35, 0.35])
    axicon.imshow(plt.imread(pic), origin = 'upper')
    axicon.set_xticks([])
    axicon.set_yticks([])
printThumbPic(imgRGBAFile)

def printBigThumb(pic):
    x0, y0 = map(ul_lon, ll_lat)
    im = plt.imread(pic)
    #x1, y1 = map(CORNER_UL_LON_PRODUCT + CORNER_UL_LON_PRODUCT - ul_lon, CORNER_LL_LAT_PRODUCT + CORNER_LL_LAT_PRODUCT - ll_lat)
    x1, y1 = map(float(CORNER_UL_LON_PRODUCT) * 2.0 - float(ul_lon), 
                 float(ll_lat) + (float(CORNER_UL_LON_PRODUCT) * 2.0 - float(ul_lon) * 2.0) * float(im.shape[1]) / float(im.shape[0]))
    y1 = y0 + float(x1 - x0) * float(im.shape[1]) / float(im.shape[0])
    plt.imshow(im, origin='upper', zorder=10, alpha=1., extent= (x0, x1, y0, y1))
#printBigThumb(imgRGBAFile)

plt.savefig(my_file.split('.')[0] + '.png')
plt.show()

