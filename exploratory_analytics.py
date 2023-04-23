#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the CSV file
data = pd.read_csv('yearwise_postcode_memberships.csv')

# Filter the data for year 2008
data_2008 = data[data['SnapshotYear'] == 2008]

# Group the data by postcode and sum the membership values
postcode_data = data_2008.groupby('Postcode')['Memberships'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
postcode_shapefile = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
postcode_data['Postcode'] = postcode_data['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
postcode_data = postcode_shapefile.merge(postcode_data, left_on='POA_CODE21', right_on='Postcode')

# Sort the data by membership intensity
postcode_data = postcode_data.sort_values(by='Memberships', ascending=False)

# Create a choropleth map
fig, ax = plt.subplots(figsize=(12, 12))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

postcode_data.plot(column='Memberships', cmap='OrRd', legend=True, ax=ax, cax=cax, legend_kwds={'label': "Memberships", 'orientation': "vertical"})

# Set the limits of the x and y axis to show the right side of Australia
ax.set_xlim(140, 155)
ax.set_ylim(-40, -25)

# Display the map
plt.show()


# In[13]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the CSV file
data = pd.read_csv('yearwise_postcode_memberships.csv')

# Filter the data for year 2008
data_2008 = data[data['SnapshotYear'] == 2017]

# Group the data by postcode and sum the membership values
postcode_data = data_2008.groupby('Postcode')['Memberships'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
postcode_shapefile = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
postcode_data['Postcode'] = postcode_data['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
postcode_data = postcode_shapefile.merge(postcode_data, left_on='POA_CODE21', right_on='Postcode')

# Sort the data by membership intensity
postcode_data = postcode_data.sort_values(by='Memberships', ascending=False)

# Create a choropleth map
fig, ax = plt.subplots(figsize=(12, 12))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

postcode_data.plot(column='Memberships', cmap='OrRd', legend=True, ax=ax, cax=cax, legend_kwds={'label': "Memberships", 'orientation': "vertical"})

# Set the limits of the x and y axis to show the right side of Australia
ax.set_xlim(140, 155)
ax.set_ylim(-40, -25)

# Display the map
plt.show()


# In[19]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the CSV file
data = pd.read_csv('postcode_memberships.csv')

# Group the data by postcode and sum the membership values
postcode_data = data.groupby('Postcode')['Memberships'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
postcode_shapefile = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
postcode_data['Postcode'] = postcode_data['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
postcode_data = postcode_shapefile.merge(postcode_data, left_on='POA_CODE21', right_on='Postcode')

# Sort the data by membership intensity
postcode_data = postcode_data.sort_values(by='Memberships', ascending=False)

# Create a choropleth map
fig, ax = plt.subplots(figsize=(12, 12))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

postcode_data.plot(column='Memberships', cmap='OrRd', legend=True, ax=ax, cax=cax, legend_kwds={'label': "Memberships", 'orientation': "vertical"})

# Set the limits of the x and y axis to show the right side of Australia
ax.set_xlim(110, 155)
ax.set_ylim(-45, -10)

# Display the map
plt.show()


# In[23]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the CSV file
data = pd.read_csv('postcode_insured_2017.csv')

# Group the data by postcode and sum the membership values
postcode_data = data.groupby('Postcode')['Memberships'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
postcode_shapefile = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
postcode_data['Postcode'] = postcode_data['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
postcode_data = postcode_shapefile.merge(postcode_data, left_on='POA_CODE21', right_on='Postcode')

# Sort the data by membership intensity
postcode_data = postcode_data.sort_values(by='Memberships', ascending=False)

# Create a choropleth map
fig, ax = plt.subplots(figsize=(12, 12))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

postcode_data.plot(column='Memberships', cmap='OrRd', legend=True, ax=ax, cax=cax, legend_kwds={'label': "Insured Persons", 'orientation': "vertical"})

# Set the limits of the x and y axis to show the right side of Australia
ax.set_xlim(140, 155)
ax.set_ylim(-40, -25)

# Display the map
plt.show()


# In[24]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the CSV file
data = pd.read_csv('postcode_claimants_anc_2017.csv')

# Group the data by postcode and sum the membership values
postcode_data = data.groupby('Postcode')['Claimants'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
postcode_shapefile = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
postcode_data['Postcode'] = postcode_data['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
postcode_data = postcode_shapefile.merge(postcode_data, left_on='POA_CODE21', right_on='Postcode')

# Sort the data by membership intensity
postcode_data = postcode_data.sort_values(by='Claimants', ascending=False)

# Create a choropleth map
fig, ax = plt.subplots(figsize=(12, 12))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

postcode_data.plot(column='Claimants', cmap='OrRd', legend=True, ax=ax, cax=cax, legend_kwds={'label': "Claimants - Ancillary Claims", 'orientation': "vertical"})

# Set the limits of the x and y axis to show the right side of Australia
ax.set_xlim(140, 155)
ax.set_ylim(-40, -25)

# Display the map
plt.show()


# In[26]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the CSV file
data = pd.read_csv('postcode_claimants_med_2017.csv')

# Group the data by postcode and sum the membership values
postcode_data = data.groupby('Postcode')['Claimants'].sum().reset_index()

# Load the shapefile
shapefile_path = 'POA_2021_AUST_GDA2020_SHP'
postcode_shapefile = gpd.read_file(shapefile_path)

# Convert Postcode column to string type
postcode_data['Postcode'] = postcode_data['Postcode'].astype(int).astype(str)

# Merge the postcode data with the postcode shapefile
postcode_data = postcode_shapefile.merge(postcode_data, left_on='POA_CODE21', right_on='Postcode')

# Sort the data by membership intensity
postcode_data = postcode_data.sort_values(by='Claimants', ascending=False)

# Create a choropleth map
fig, ax = plt.subplots(figsize=(12, 12))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

postcode_data.plot(column='Claimants', cmap='OrRd', legend=True, ax=ax, cax=cax, legend_kwds={'label': "Claimants - Medical Claims", 'orientation': "vertical"})

# Set the limits of the x and y axis to show the right side of Australia
ax.set_xlim(140, 155)
ax.set_ylim(-40, -25)

# Display the map
plt.show()


# In[ ]:




