#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import pysal.lib.weights as weights
import pysal.explore.esda as esda
import matplotlib.pyplot as plt
import splot.esda as sp

# Set plot style
plt.style.use('seaborn-whitegrid')

# Load the dataset into a pandas dataframe
df = pd.read_csv('mental_health_year_postcode_cost.csv')

# Group the data by postcode and sum the membership values
df = df.groupby('Postcode')['MentalHealthCost'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
aus_postcodes = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
df['Postcode'] = df['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
gdf = aus_postcodes.merge(df, left_on='POA_CODE21', right_on='Postcode')

# Keep only the relevant columns
gdf = gdf[['Postcode', 'geometry','MentalHealthCost']]

# Keep only geometries with type 'MultiPolygon'
gdf = gdf[gdf.geometry.type == 'Polygon']

gdf

# Create a map of the data
gdf.plot(column='MentalHealthCost', cmap='coolwarm', legend=True)
plt.title('Mental Health Cost by Postcode')
plt.show()

# Extract the coordinates from the polygon centroids
coords = gdf.geometry.centroid.apply(lambda p: (p.x, p.y)).tolist()

# Create a spatial weights matrix based on the coordinates
w = weights.Kernel(coords, bandwidth=1000)

# Standardize the MentalHealthCost variable
gdf['MentalHealthCost_std'] = (gdf['MentalHealthCost'] - gdf['MentalHealthCost'].mean()) / gdf['MentalHealthCost'].std()

# Calculate Moran's I statistic
moran = esda.Moran(gdf['MentalHealthCost_std'], w)

# Print the Moran's I statistic and p-value
print("Moran's I:", moran.I)
print("p-value:", moran.p_sim)

# Calculate Getis-Ord statistics
g_local = esda.G_Local(gdf['MentalHealthCost_std'], w)

print('Getis-Ord Gi* statistic:')
print(g_local.p_sim)

# Create a Local Moran's I plot
from pysal.viz.splot.esda import moran_scatterplot
fig, ax = moran_scatterplot(moran, p=0.05, aspect_equal=True)
plt.show()


# In[4]:


import pandas as pd
import geopandas as gpd
import pysal.lib.weights as weights
import pysal.explore.esda as esda
import matplotlib.pyplot as plt
import splot.esda as sp

# Set plot style
plt.style.use('seaborn-whitegrid')

# Load the dataset into a pandas dataframe
df = pd.read_csv('addiction_year_postcode_cost.csv')

# Group the data by postcode and sum the membership values
df = df.groupby('Postcode')['AddictionCost'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
aus_postcodes = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
df['Postcode'] = df['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
gdf = aus_postcodes.merge(df, left_on='POA_CODE21', right_on='Postcode')

# Keep only the relevant columns
gdf = gdf[['Postcode', 'geometry','AddictionCost']]

# Keep only geometries with type 'MultiPolygon'
gdf = gdf[gdf.geometry.type == 'Polygon']

# Create a map of the data
gdf.plot(column='AddictionCost', cmap='coolwarm', legend=True)
plt.title('Addiction Cost by Postcode')
plt.show()

# Extract the coordinates from the polygon centroids
coords = gdf.geometry.centroid.apply(lambda p: (p.x, p.y)).tolist()

# Create a spatial weights matrix based on the coordinates
w = weights.Kernel(coords, bandwidth=1000)

# Standardize the AddictionCost variable
gdf['AddictionCost_std'] = (gdf['AddictionCost'] - gdf['AddictionCost'].mean()) / gdf['AddictionCost'].std()

# Calculate Moran's I statistic
moran = esda.Moran(gdf['AddictionCost_std'], w)

# Print the Moran's I statistic and p-value
print("Moran's I:", moran.I)
print("p-value:", moran.p_sim)

# Calculate Getis-Ord statistics
g_local = esda.G_Local(gdf['AddictionCost_std'], w)

print('Getis-Ord Gi* statistic:')
print(g_local.p_sim)

# Create a Local Moran's I plot
from pysal.viz.splot.esda import moran_scatterplot
fig, ax = moran_scatterplot(moran, p=0.05, aspect_equal=True)
plt.show()


# In[6]:


import pandas as pd
import geopandas as gpd
import pysal.lib.weights as weights
import pysal.explore.esda as esda
import matplotlib.pyplot as plt
import splot.esda as sp

# Set plot style
plt.style.use('seaborn-whitegrid')

# Load the dataset into a pandas dataframe
df = pd.read_csv('obesity_year_postcode_cost.csv')

# Group the data by postcode and sum the membership values
df = df.groupby('Postcode')['ObesityCost'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
aus_postcodes = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
df['Postcode'] = df['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
gdf = aus_postcodes.merge(df, left_on='POA_CODE21', right_on='Postcode')

# Keep only the relevant columns
gdf = gdf[['Postcode', 'geometry','ObesityCost']]

# Keep only geometries with type 'MultiPolygon'
gdf = gdf[gdf.geometry.type == 'Polygon']

# Create a map of the data
gdf.plot(column='ObesityCost', cmap='coolwarm', legend=True)
plt.title('Obesity Cost by Postcode')
plt.show()

# Extract the coordinates from the polygon centroids
coords = gdf.geometry.centroid.apply(lambda p: (p.x, p.y)).tolist()

# Create a spatial weights matrix based on the coordinates
w = weights.Kernel(coords, bandwidth=1000)

# Standardize the ObesityCost variable
gdf['ObesityCost_std'] = (gdf['ObesityCost'] - gdf['ObesityCost'].mean()) / gdf['ObesityCost'].std()

# Calculate Moran's I statistic
moran = esda.Moran(gdf['ObesityCost_std'], w)

# Print the Moran's I statistic and p-value
print("Moran's I:", moran.I)
print("p-value:", moran.p_sim)

# Calculate Getis-Ord statistics
g_local = esda.G_Local(gdf['ObesityCost_std'], w)

print('Getis-Ord Gi* statistic:')
print(g_local.p_sim)

# Create a Local Moran's I plot
from pysal.viz.splot.esda import moran_scatterplot
fig, ax = moran_scatterplot(moran, p=0.05, aspect_equal=True)
plt.show()


# In[ ]:


import pandas as pd
import geopandas as gpd
import pysal.lib.weights as weights
import pysal.explore.esda as esda
import matplotlib.pyplot as plt
import splot.esda as sp

# Set plot style
plt.style.use('seaborn-whitegrid')

# Load the dataset into a pandas dataframe
df = pd.read_csv('musculoskeletal_diseases_year_postcode_cost.csv')

# Group the data by postcode and sum the membership values
df = df.groupby('Postcode')['MusculoskeletalDiseasesCost'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
aus_postcodes = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
df['Postcode'] = df['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
gdf = aus_postcodes.merge(df, left_on='POA_CODE21', right_on='Postcode')

# Keep only the relevant columns
gdf = gdf[['Postcode', 'geometry','MusculoskeletalDiseasesCost']]

# Keep only geometries with type 'MultiPolygon'
gdf = gdf[gdf.geometry.type == 'Polygon']

# Create a map of the data
gdf.plot(column='MusculoskeletalDiseasesCost', cmap='coolwarm', legend=True)
plt.title('Musculoskeletal Diseases Cost by Postcode')
plt.show()

# Extract the coordinates from the polygon centroids
coords = gdf.geometry.centroid.apply(lambda p: (p.x, p.y)).tolist()

# Create a spatial weights matrix based on the coordinates
w = weights.Kernel(coords, bandwidth=1000)

# Standardize the MusculoskeletalDiseasesCost variable
gdf['MusculoskeletalDiseasesCost_std'] = (gdf['MusculoskeletalDiseasesCost'] - gdf['MusculoskeletalDiseasesCost'].mean()) / gdf['MusculoskeletalDiseasesCost'].std()

# Calculate Moran's I statistic
moran = esda.Moran(gdf['MusculoskeletalDiseasesCost_std'], w)

# Print the Moran's I statistic and p-value
print("Moran's I:", moran.I)
print("p-value:", moran.p_sim)

# Calculate Getis-Ord statistics
g_local = esda.G_Local(gdf['MusculoskeletalDiseasesCost_std'], w)

print('Getis-Ord Gi* statistic:')
print(g_local.p_sim)

# Create a Local Moran's I plot
from pysal.viz.splot.esda import moran_scatterplot
fig, ax = moran_scatterplot(moran, p=0.05, aspect_equal=True)
plt.show()

