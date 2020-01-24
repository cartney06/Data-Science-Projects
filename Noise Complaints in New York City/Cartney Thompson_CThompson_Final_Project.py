import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance
from datetime import datetime
import matplotlib.pyplot as plt

#Load in data file
complaints = pd.read_csv('party_in_nyc.csv')

#Load in Census files
census = pd.read_csv('nyc_census_tracts.csv') #For census data
block = pd.read_csv('census_block_loc.csv') #For census block lookup
censusTract = pd.read_csv('ZIP_TRACT_122017.csv') #For zip to census tract lookup

########################################################## Data Preprocessing #############################################################################################
########## Complaint Data ################################
#Create an id column for each complaint
complaints["id"] = complaints.index + 1

#Convert time stamps from strings to datetime
complaints['Created Date'] =  pd.to_datetime(complaints['Created Date'], format = '%Y-%m-%d %H:%M:%S')
complaints['Closed Date'] =  pd.to_datetime(complaints['Closed Date'], format = '%Y-%m-%d %H:%M:%S')

#Check for Na's
complaints.isna().sum()

#Create dataframe with rows that do not have Incident Zip with NAs
complaints = complaints[np.isfinite(complaints['Incident Zip'])]

#Create dataframe with rows that do not have Incident Zip with NAs
complaints = complaints[np.isfinite(complaints['Latitude'])]

#Keep only 2016 complaints
complaints = complaints[ complaints['Created Date']>='2016-01-01']

########## Census Block Data #############################
#Only select counties that belong to NYC (Kings, Queens, Bronx, Richmond, New York)
block = block.loc[block['County'].isin(['Richmond', 'New York', 'Kings', 'Queens', 'Bronx'])]

#Convert BlockCode to str as a new columns
block.BlockCode = block.BlockCode.astype(str)


########## Census Data #############################
#Remove Na's from census data
census = census.dropna()

#Convert CensusTract to str
census.CensusTract = census.CensusTract.astype(str)

#########################################################
#Merge census and census block data based upon Censustract
censusBlock = census

block['CensusTract'] = block['BlockCode'].str[:11]

#Merge censusBlock with census columns 
censusMerged = pd.merge(censusBlock, block, how='left', on='CensusTract')

#Copy of complaints dataframe into five parts
df1 = complaints[:50000]
df2 = complaints[50001:100000]
df3 = complaints[100001:150000]
df4 = complaints[150001:200000]
df5 = complaints[200001:250000]

#Create distance calculation using scipy. We will split the computations up to limit memory requirements
mat1 = scipy.spatial.distance.cdist(df1[['Latitude','Longitude']], censusMerged[['Latitude','Longitude']], metric='euclidean')
mat2 = scipy.spatial.distance.cdist(df2[['Latitude','Longitude']], censusMerged[['Latitude','Longitude']], metric='euclidean')
mat3 = scipy.spatial.distance.cdist(df3[['Latitude','Longitude']], censusMerged[['Latitude','Longitude']], metric='euclidean')
mat4 = scipy.spatial.distance.cdist(df4[['Latitude','Longitude']], censusMerged[['Latitude','Longitude']], metric='euclidean')
mat5 = scipy.spatial.distance.cdist(df5[['Latitude','Longitude']], censusMerged[['Latitude','Longitude']], metric='euclidean')

#Convert new distance calcultaions with into dataframe
new_df1 = pd.DataFrame(mat1, index=df1['id'], columns=censusMerged['BlockCode']) 
new_df2 = pd.DataFrame(mat2, index=df2['id'], columns=censusMerged['BlockCode']) 
new_df3 = pd.DataFrame(mat3, index=df3['id'], columns=censusMerged['BlockCode'])
new_df4 = pd.DataFrame(mat4, index=df4['id'], columns=censusMerged['BlockCode']) 
new_df5 = pd.DataFrame(mat5, index=df5['id'], columns=censusMerged['BlockCode']) 

#Get nearest censusblock based upon min distance for each complaint
blah1 = new_df1.idxmin(axis=1)
blah2 = new_df2.idxmin(axis=1)
blah3 = new_df3.idxmin(axis=1)
blah4 = new_df4.idxmin(axis=1)
blah5 = new_df5.idxmin(axis=1)

#Convert above to a dataframe
blah1 = pd.DataFrame({'id':blah1.index, 'NearestBlockCode':blah1.values})
blah2 = pd.DataFrame({'id':blah2.index, 'NearestBlockCode':blah2.values})
blah3 = pd.DataFrame({'id':blah3.index, 'NearestBlockCode':blah3.values})
blah4 = pd.DataFrame({'id':blah4.index, 'NearestBlockCode':blah4.values})
blah5 = pd.DataFrame({'id':blah5.index, 'NearestBlockCode':blah5.values})

#Concatenate to a single dataframe
frames = pd.concat([blah1, blah2, blah3, blah4, blah5])

#Drop unneccessary columns from censusMerged dataframe
censusMerged = censusMerged.drop(['Latitude', 'Longitude', 'County_y', 'State'], axis=1)
#Delete - Remove Duplicates
censusMerged = censusMerged.drop_duplicates(subset=(['BlockCode', 'CensusTract']))

#Merge complaint with nearest censusblock data
mergedData = pd.merge(complaints, frames, on='id')

#Final merge with Census Data
mergedData = pd.merge(mergedData, censusMerged, how='left', left_on='NearestBlockCode', right_on='BlockCode')

########################################################## Data Exploration #############################################################################################

#### Time Series ######
#Counting the number of unique id's will give us  count of total number of complaints
totalComplaints = mergedData['id'].nunique()

#Count by month, number of complaints (Pivot Table?)
monthCount = mergedData['id'].groupby([mergedData['Created Date'].dt.year.rename('year'), mergedData['Created Date'].dt.month.rename('month')]).agg({'count'})

#Count by day, number of complaints (Pivot Table?)
dayCount = mergedData['id'].groupby([mergedData['Created Date'].dt.date.rename('date')]).agg({'count'})


#Percentage of Noise Complaints by month and Export
monthPct = (monthCount/monthCount.sum())

#Ranking the top 5 days in terms of noise complaints
dayCount.sort_values('count', ascending=False).head(5)

#Count by hour, number of complaints (Pivot Table?)
hourCount = mergedData['id'].groupby([mergedData['Created Date'].dt.hour.rename('Hour')]).agg({'count'})

#Percentage of Noise Complaints by hour and Export
hourPct = (hourCount/hourCount.sum())

#Noise complaints by day of week
weekdayCount = mergedData['id'].groupby([mergedData['Created Date'].dt.weekday_name.rename('day')]).agg({'count'})

#Re-Index days of week
weekdayCount = weekdayCount.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

#Percentage of Noise Complaints by day of week and Export
weekdayPct = (weekdayCount/weekdayCount.sum())

### Location Type ###
#Percentage of Noise Complaints by Location Type & Export
locationTypeCount = mergedData['id'].groupby([mergedData['Location Type'].rename('locationType')]).agg({'count'})
locationTypePct = (locationTypeCount/locationTypeCount.sum()).sort_values('count', ascending=False)

### By Location ###
#Noise complaints by borough
boroughCount = mergedData['id'].groupby([mergedData['Borough_x'].rename('Borough')]).agg({'count'})
boroughCountPct = (boroughCount/boroughCount.sum()).sort_values('count', ascending=False) 
#Add in population data for each borough - Source: United States Census Bureau - 2018
boroughCountPct['populationPct'] = [ 2648771/8622698, 1664727/8622698, 1471160/8622698, 2358582/8622698, 479458/8622698]
#Calculate index for likelihood of submitting a noise complaint and export the result
boroughCountPct['index'] = boroughCountPct['count']/boroughCountPct['populationPct']

#Top 10 census tracts for noise complaints and their demographics
#Noise complaints by census block
censusTractCount = mergedData['id'].groupby([mergedData['CensusTract'].rename('CensusTract')]).agg({'count'}).reset_index(drop=False)
#Join censusTractCount with census data
censusTractCount = pd.merge(censusTractCount, census, how='left', on='CensusTract')

#Rename the count column to complaints
censusTractCount.rename(columns = {'count':'Complaints'}, inplace = True)

#Add a column ComplaintsPerCapita to get Noise complaints per capita
censusTractCount['ComplaintsPerCapita'] = (censusTractCount['Complaints']/censusTractCount['TotalPop'])

#Get top 10 census tracts based upon noise complaints per capita
censusTractCountTop10 = censusTractCount.sort_values('ComplaintsPerCapita', ascending=False).head(10)

#Export the results to a .xlsx file
censusTractCountTop10 = censusTractCountTop10[['CensusTract', 'County', 'Borough',  'Complaints', 'ComplaintsPerCapita', 'TotalPop', 'Hispanic', 'Black', 'Asian' ,'White', 'Income']]

fname = 'NoiseComplaintsOutput.xlsx'
writer = pd.ExcelWriter(fname)
monthPct.to_excel(writer, "MonthPct", index=True)
hourPct.to_excel(writer, "HourPct", index=True)
weekdayPct.to_excel(writer, "WeekdayPct", index=True)
locationTypePct.to_excel(writer, "LocationTypePct", index=True)
boroughCountPct.to_excel(writer, "BoroughCountPct", index=True)
censusTractCountTop10.to_excel(writer, "CensusTractTop10", index=False)
writer.save()

print('Total number of complaints in NYC during 2016:', totalComplaints)
print('File completed. Please check your output folder!')