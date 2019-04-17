
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import math

file1 = pd.read_csv('file1.csv', ',')

file1.describe()


# In[2]:


plt.scatter(file1['accuracy'],file1['speedkmh'])
plt.xlabel('Accuracy (meters)')
plt.ylabel('Speed (kph)')
plt.title('Speed vs. Accuracy (Raw Data)')

plt.plot(np.unique(file1['accuracy']),          np.poly1d(np.polyfit(file1['accuracy'], file1['speedkmh'], 1))         (np.unique(file1['accuracy'])))
#There is a positive correlation between speed and accuracy.


# In[3]:


sns.distplot(file1['speedkmh'], kde = False)
plt.title('Speed Distribution (Raw Data)')
plt.ylabel('Count')
plt.xlabel('Speed (kph)')


# In[4]:


mean_speedkmh = file1['speedkmh'].mean()
#The average speed for this dataset is 63.29 kph.

std_speedkmh = file1['speedkmh'].std()
#29.5 kph
upperbound_2sig = mean_speedkmh + 2 * std_speedkmh
#122.3 kph ~ 76 mph
upperbound_3sig = mean_speedkmh + 3 * std_speedkmh
#151.8 kph ~ 94 mph


# In[5]:


sns.distplot(file1['accuracy'], kde = False)
plt.title('Accuracy Distribution (Raw Data)')
plt.xlabel('Accuracy (m)')
plt.ylabel('Count')


# In[6]:


file1_clean = file1.loc[(file1['speedkmh'] < upperbound_2sig) &                         (file1['accuracy'] <= 10)]

file1_clean.describe()


# In[7]:


sns.distplot(file1_clean['speedkmh'], kde = False)
plt.title('Speed Distribution (Clean)')
plt.xlabel('Speed (kph)')
plt.ylabel('Count')


# In[8]:


plt.scatter(file1_clean['accuracy'],file1_clean['speedkmh'])

plt.plot(np.unique(file1_clean['accuracy']),          np.poly1d(np.polyfit(file1_clean['accuracy'], file1_clean['speedkmh'], 1))         (np.unique(file1_clean['accuracy'])))

plt.title('Speed vs. Accuracy (Clean)')
plt.xlabel('Accuracy (kph)')
plt.ylabel('Speed (kph)')


#There is clumping present at 5 m and 10 m accuracy, though there is no longer any 
#correlation in this plot between speed and accuracy. This is ideal because we want 
#speed and accuracy to act as independent variables. 


# In[9]:


file1_clean_acc10 = file1_clean.loc[(file1['accuracy'] == 10.0)]
file1_clean_acc5 = file1_clean.loc[(file1['accuracy'] == 5.0)]

file1_clean_acc10.describe()


# In[10]:


file1_clean_acc5.describe()


# In[12]:


file1_clean['Coordinates'] = list(zip(file1_clean.longitude, file1_clean.latitude))

file1_clean['Coordinates'] = file1_clean['Coordinates'].apply(Point)

geofile1 = gpd.GeoDataFrame(file1_clean, geometry='Coordinates')

usa = gpd.read_file('tl_2018_us_state/tl_2018_us_state.shp')
coastline = gpd.read_file('tl_2018_us_coastline/tl_2018_us_coastline.shp')
laporte_in = gpd.read_file('tl_2018_18091_roads/tl_2018_18091_roads.shp')
lake_in = gpd.read_file('tl_2018_18089_roads/tl_2018_18089_roads.shp')
porter_in = gpd.read_file('tl_2018_18127_roads/tl_2018_18127_roads.shp')
cook_il = gpd.read_file('tl_2018_17031_roads/tl_2018_17031_roads.shp')
berrian_mi = gpd.read_file('tl_2018_26021_roads/tl_2018_26021_roads.shp')

#https://www.census.gov/cgi-bin/geo/shapefiles/index.php?

fig, ax = plt.subplots(figsize = (30,30))
ax.set_xlim([-87.75, -86.6])
ax.set_ylim([41.5, 42.0])

usa[usa.NAME == 'Indiana'].plot(ax = ax, color='white', edgecolor='orange')
usa[usa.NAME == 'Illinois'].plot(ax = ax, color='white', edgecolor='orange')
laporte_in.plot(ax = ax, alpha = 0.2)
lake_in.plot(ax = ax, alpha = 0.2)
porter_in.plot(ax = ax, alpha = 0.2)
cook_il.plot(ax = ax, alpha = 0.2)
berrian_mi.plot(ax = ax, alpha = 0.2)
coastline.plot(ax = ax)

#Chicago, IL to La Porte, IN

geofile1.plot(ax=ax, color='red', alpha = 0.5)

plt.title('Geolocation Map of Illinois and Indiana Leg of Cross-Country Roadtrip', fontsize=20)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.show()


# In[13]:


#https://www.thoughtco.com/degree-of-latitude-and-longitude-distance-4070616
#Latitude: 1 deg = 111 km
#Longitude: 1 deg = 85 km

start = (-86.75528299999999, 41.603271)
end = (-87.638186, 41.872075)

delta_long = start[0] - end[0]
delta_lat = start[1] - end[1]

delta_km_ew = delta_long * 85
delta_km_ns = delta_lat * 111

distance = (delta_km_ew**2 + delta_km_ns**2)**0.5

print(distance)


# In[14]:


total_distance = 0
for x in range(2635):
    start = (file1_clean.iloc[x,2], file1_clean.iloc[x,1])
    end = (file1_clean.iloc[x+1,2], file1_clean.iloc[x+1,1])
    
    delta_long = start[0] - end[0]
    delta_lat = start[1] - end[1]

    delta_km_ew = delta_long * 85
    delta_km_ns = delta_lat * 111

    distance = (delta_km_ew**2 + delta_km_ns**2)**0.5
    
    total_distance = total_distance + distance
print(total_distance)


# In[ ]:




