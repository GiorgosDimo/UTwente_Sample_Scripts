import pandas as pd 
import datetime
from collections import defaultdict, Counter
from scipy.signal import savgol_filter
import numpy as np

#Load the excel sheet that we need
data = pd.read_excel(r"2015-2016_Tekenvangsten_Natuurkalender.xlsx", sheet_name='2015-2016') 
#Change the name of the location De Kwode Hoek and Hoog Baarlo in order to be the same as in the case of Irene's inputs
for l in range(len(data['Location'])):
    if data.at[l,'Location'] == 'De Kwade Hoek':
        data.at[l,'Location'] = 'KwadeHoek'
        
    elif data.at[l,'Location'] == 'Hoog Baarlo':
        data.at[l,'Location'] = 'HoogBaarlo' 
#Add the index of the row as a column (rowid) 
data['rowid'] = data.index        
#combine the location and the plot number in order to have the same names as in the case of Irene's inputs
data['name'] = data['Location'] + '_' + data['Plot number'].astype(str)
#Remove the columns that we do not need and keep only the name, the date and the Nimph columns.
data = data.drop(['Collection month','Location','Plot number', 'Week number', 'Day', 'Month', 'Year', 'Male', 'Female', 'Larvae', 'Remark '], 1)
#Re-arrange the columns in the order that we need
data = data[['rowid', 'name', 'Date', 'Nimph']]
data.rename(columns={'Nimph': 'TickCount'}, inplace=True)
#Remove all the rows that have nan values for the Nimphs
data.dropna(subset = ["TickCount"], inplace=True)
#Remove the Vaals_2 since it has only one value
indexNames = data[(data['name'] == 'Vaals_2')].index 
data.drop(indexNames , inplace=True)
#Keep only the date from the timestamp
pd.DatetimeIndex(data.Date).normalize()
data['Date'] = data['Date'].dt.date
#Create and polpulate a dictionary with all the values for each available name
original_ticks = defaultdict(list)
for i in range(len(data['name'])):
    rowid = data['rowid'].iloc[i]
    Dates = data['Date'].iloc[i]
    Tick_count = data['TickCount'].iloc[i]
    original_ticks[data['name'].iloc[i]].append([rowid, Dates, Tick_count])

smoothed_ticks = defaultdict(list)
for key in sorted(original_ticks.keys()):
    rowid = np.array([item[0] for item in original_ticks[key]])
    dates = np.array([item[1] for item in original_ticks[key]]).astype(str)
    target = np.array([item[2] for item in original_ticks[key]])
    fit = savgol_filter(target, 7, 5)
    smoothed_ticks[key].append(np.stack((rowid, dates, fit)).transpose())

smoothed_ticks_array = np.empty(shape = (len(data)+1, 4), dtype='object')
n = 0
for key in sorted(smoothed_ticks.keys()):
    for i in range(len(smoothed_ticks[key][0])):
        rowid = smoothed_ticks[key][0][i][0]
        date = smoothed_ticks[key][0][i][1]
        fit = smoothed_ticks[key][0][i][2]
        n += 1
        #print(n)
        smoothed_ticks_array[n][0] = rowid
        smoothed_ticks_array[n][1] = key
        smoothed_ticks_array[n][2] = date
        smoothed_ticks_array[n][3] = round(float(fit), 2)

smoothed_ticks_pd = pd.DataFrame({'rowid': smoothed_ticks_array[:, 0], 'name':smoothed_ticks_array[:, 1], 'date':smoothed_ticks_array[:, 2], 'tickCount':smoothed_ticks_array[:, 3]})

smoothed_ticks_pd.to_csv('volunteer_data_savgol_2015-2016.csv')
'''
#Taking a list of dates
def_dic_dates = defaultdict(list)
for d in range(len(data['Date'])):
    #def_dic_dates[data['Date'].iloc[d]].append(data['name'].iloc[d])
    def_dic_dates[data['name'].iloc[d]].append(data['Date'].iloc[d])
#Appelscha_1_smooth = savgol_filter(def_dic['Appelscha_1'], 5, 2)
#Ede_1_smooth = savgol_filter(def_dic['Ede_1'], 5, 2)

dates=[]
for item in def_dic_dates:
    dates.append(str(item))
'''


