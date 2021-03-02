import ee
from geetools import batch
import pandas as pd
from collections import defaultdict
import datetime
from functools import reduce

try:
  ee.Initialize()
  print('Google Earth Engine has initialized successfully!')
except ee.EEException as e:
  print('Google Earth Engine has failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise


# take the mean of the vegetation indices per month for reducing the impact of the clouds and add a timestamp and the imgnumber for refference
def meanVI(n, modis, div, epoch):
    date = epoch.advance(n, 'month')
    start = date.advance(-1, 'day')
    end = date.advance(1, 'month').advance(-1, 'day')
    #divide the values by 10000 in NDVI and EVI values for proper scaling 
    mosaic = modis.filterDate(start, end).mean().divide(div)
    mosaic = mosaic.addBands((ee.Image.constant(start.millis()).divide(1000 * 60 * 60 * 24 * 365)).rename('timestamp'))
    mosaic = mosaic.addBands(ee.Image.constant(n).rename('imgnumber'))
    return mosaic;
# add the coordinates to the image collection while reducing it to only the flagging sites
def addlatlon(img):
    return img.addBands(ee.Image.pixelLonLat()).reduceRegions(flagging, 'first', 1, crs)

# a fuction to create the raster images that we need 
def createMinRangeImages(year):
    
    #define starting date
    epoch = ee.Date(year + '-01-01')
    #define number of months until the end of the year in question 
    N = 11
    # apply the above function for the period in question
    images_ndvi_evi = ee.List.sequence(0, N).map(lambda x : meanVI(x, modis_ndvi_evi, 10000, epoch))
    images_ndwi = ee.List.sequence(0, N).map(lambda x : meanVI(x, modis_ndwi, 1, epoch))
    # create an image collection
    collection_ndvi_evi = ee.ImageCollection.fromImages(images_ndvi_evi)
    collection_ndwi = ee.ImageCollection.fromImages(images_ndwi)
    
    # extract the images for the minimum/maximum/range value per year per VI
    minNDVI = (collection_ndvi_evi.select('NDVI')).min()
    maxNDVI = (collection_ndvi_evi.select('NDVI')).max()
    rangeNDVI = maxNDVI.subtract(minNDVI)
    minEVI = (collection_ndvi_evi.select('EVI')).min()
    maxEVI = (collection_ndvi_evi.select('EVI')).max()
    rangeEVI = maxEVI.subtract(minEVI)
    minNDWI = (collection_ndwi.select('NDWI')).min()
    
    return [minNDVI.addBands(ee.Image.constant(int(year)).rename('timestamp')), 
            rangeNDVI.addBands(ee.Image.constant(int(year)).rename('timestamp')), 
            minEVI.addBands(ee.Image.constant(int(year)).rename('timestamp')), 
            rangeEVI.addBands(ee.Image.constant(int(year)).rename('timestamp')), 
            minNDWI.addBands(ee.Image.constant(int(year)).rename('timestamp'))]

# Convert the dictionary into pandas dataframe
def todatafarme(dict_result):
    result_df = pd.DataFrame()
    for dist in dict_result['features']:
        df = pd.DataFrame([dist['properties']], columns=dist['properties'].keys())
        result_df = pd.concat([result_df, df], axis=0)
    return result_df

#take the location from the coordinates 
def addLocation (row, dictionary):
    lat_long = str(round(row['latitude'], 4))+'_'+ str(round(row['longitude'], 4))
    return dictionary[lat_long]

#
def ImagesToDataframe(year, flagging):
    #compute the min amd range values of VI
    VIs = createMinRangeImages(year)
    
    #extract the VIs for every flagging location and add the coordinates as bands 
    VIs_Fl = list(map(lambda x: x.addBands(ee.Image.pixelLonLat()).reduceRegions(flagging, 'first', 1, 'EPSG:4326'), VIs))
    VIs_dicts = list(map(lambda x: x.select([".*"], retainGeometry=False).getInfo(), VIs_Fl))
    result_dfs = list(map(todatafarme, VIs_dicts))
    
    df_final = reduce(lambda left,right: pd.merge(left,right,on=['latitude', 'longitude', 'timestamp']), result_dfs)
    #remove duplicates
    df_final = df_final.drop_duplicates(subset ="latitude")
    # rename and re-arrange the columns
    df_final.rename(columns={'NDVI_x': 'min_ndvi', 'NDVI_y': 'range_ndvi', 'EVI_x': 'min_evi', 'EVI_y': 'range_evi', 'NDWI': 'min_ndwi'}, inplace=True)
    df_final = df_final[['latitude', 'longitude', 'timestamp', 'min_ndvi', 'range_ndvi', 'min_evi', 'range_evi', 'min_ndwi']]
    #devide the df_final into the single and double coordinates per location in order to add the location 
    df_single = df_final.head(len(single_coord_list))
    df_double = df_final.tail(len(double_coord_list)*2)
    # reverse the location dictionaries 
    flagging_dict_singe_reverse = {str(round(v[1],4))+'_'+str(round(v[0],4)) : k for k, v in flagging_dict_single.items()}
    flagging_dict_double_reverse = {str(round(v[1],4))+'_'+str(round(v[0],4)) : k for k, v in flagging_dict_double.items()}
    
    #add the location on the dataframes   
    df_single['location'] = df_single.apply (lambda row: addLocation(row, flagging_dict_singe_reverse), axis=1)
    df_double['location'] = df_double.apply (lambda row: addLocation(row, flagging_dict_double_reverse), axis=1)
    #copy the simgle dataframe in order to have 2 transects and change the location value
    df_single_2 = df_single.copy()
    df_single_2['location'] = df_single_2['location'].str.replace('2','1')
    df_single_final = df_single_2.append(df_single)
    #combine the dataframes with the locations 
    result_df = df_single_final.append(df_double)
    result_df = result_df.sort_values('location')
    #keep only the columns that we need
    result_df = result_df[['location', 'timestamp', 'min_ndvi', 'range_ndvi', 'min_evi', 'range_evi', 'min_ndwi']]
    return result_df

################
# Main Program #
################
#Inputs:
#import the datasets that we need
modis_ndwi = ee.ImageCollection("MODIS/MCD43A4_006_NDWI")
modis_ndvi_evi = ee.ImageCollection('MODIS/006/MYD13Q1').select(['NDVI', 'EVI']);
nl = ee.Image('users/georgosdimopoulos/nederland_1000')
#extract the crs from the dataset 
crs = ee.Image(modis_ndvi_evi.first()).projection().crs()
#extract the crs parameters from the target  
crs_nl = nl.projection().crs()
transform = [1000, 0, 13565.399999997899, 0, -1000, 619315.6350000015]
# import the shapfile as feature collection with the sampling locations and their coordinates in EPSG:4326
sampl = ee.FeatureCollection('users/georgosdimopoulos/flagging_sites_DMS_only_complete_ts')
#select the locations that have for both transects the same coordinates
single_coord_list = ['Nijverdal', 'Ede', 'Hoog Baarlo', 'Twiske', 'Kwade Hoek', 'Veldhoven']
#select the locations that have different coordinates per transect
double_coord_list = ['Schiermonnikoog','Appelscha', 'Gieten', 'Montferland', 'Bilthoven', 'Dronten', 'Wassenaar', 'Eijsden', 'Vaals']
#create similar lists with the proper names that we need for the final table
name_list_s = ['Nijverdal_1', 'Nijverdal_2', 'Ede_1', 'Ede_2', 'HoogBaarlo_1', 'HoogBaarlo_2', 'Twiske_1', 'Twiske_2', 
                     'KwadeHoek_1', 'KwadeHoek_2', 'Veldhoven_1', 'Veldhoven_2']
name_list_d = ['Schiermonnikoog_1', 'Schiermonnikoog_2', 'Appelscha_1', 'Appelscha_2', 'Gieten_1', 'Gieten_2', 'Montferland_1', 'Montferland_2', 
                     'Bilthoven_1', 'Bilthoven_2', 'Dronten_1', 'Dronten_2', 'Wassenaar_1', 'Wassenaar_2', 'Eijsden_1', 'Eijsden_2', 'Vaals_1', 'Vaals_2']    

#take only the locations that we need from the whole feature collection
filter_single = ee.Filter.inList('location', single_coord_list);
filter_double = ee.Filter.inList('location', double_coord_list);

filtered_single = sampl.filter(filter_single);
filtered_double = sampl.filter(filter_double);
#extract the coordinates for every locations and transform it into a list
filtered_single_coord = ((filtered_single.geometry()).coordinates()).getInfo()
filtered_double_coord = ((filtered_double.geometry()).coordinates()).getInfo()
#double the values of the locations with the single coordinates pair 
doubled = list(zip(filtered_single_coord,filtered_single_coord)) 
doubled_single_coord = [item for sublist in list(doubled) for item in sublist]
#create the dictionaries and combine them into one
flagging_dict_single = dict(zip(name_list_s, doubled_single_coord))
flagging_dict_double = dict(zip(name_list_d, filtered_double_coord))
flagging_dict = {**flagging_dict_single, **flagging_dict_double}  
    

#create a list with the coordinates in gee geometry points  
flagging_sites = []
for key in flagging_dict:
    flagging_sites.append(ee.Geometry.Point(flagging_dict[key]))

#create a feature collection from the points
flagging = ee.FeatureCollection(flagging_sites)

#define the year that we need to compute the VIs
years = [str(i) for i in range(2015,2017,1)]

#compute dataframe for export to csv
df_per_year = list(map(lambda x: ImagesToDataframe(x, flagging), years))
df_for_csv = pd.concat(df_per_year)
#export to csv
#df_for_csv.to_csv('VI_predictors_2006-2014.csv', index=False)
df_for_csv.to_csv('VI_predictors_2015-2016_new.csv', index=False)


