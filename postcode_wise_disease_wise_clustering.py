#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pgeocode
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import folium


# In[2]:


# importing the dataset
mentalHealth_df = pd.read_csv('postcode_wise/postcode_wise_mental_health_cost.csv')
addiction_df = pd.read_csv('postcode_wise/postcode_wise_addiction_cost.csv')
obesity_df = pd.read_csv('postcode_wise/postcode_wise_obesity_cost.csv')
musculoskeletal_df = pd.read_csv('postcode_wise/postcode_wise_musculoskeletal_diseases_cost.csv')
print(len(mentalHealth_df))
print(len(addiction_df))
print(len(obesity_df))
print(len(musculoskeletal_df))


# In[3]:


mentalHealth_df.head()


# In[4]:


addiction_df.head()


# In[5]:


obesity_df.head()


# In[6]:


musculoskeletal_df.head()


# In[7]:


result1 = pd.merge(mentalHealth_df, addiction_df, how="inner", on=["Postcode", "Postcode"])
result2 = pd.merge(obesity_df, musculoskeletal_df, how="inner", on=["Postcode", "Postcode"])
#df = pd.merge(result1, result2, how="inner", on=["Postcode", "Postcode"])
df = result1
#df = result2


# In[8]:


print(len(df))


# In[9]:


# removing any NAN and inf values from the dataframe
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
df


# In[10]:


print(len(df))


# In[11]:


df.head()


# In[12]:


original_df = df
original_df.head()


# In[13]:


#df = df[(df['FinancialYear'] == 2017)]
#df.head()


# In[14]:


#df = df.sample(n=1000,random_state=1)


# In[15]:


# convert postcode column to integer data type
df = df.astype({'Postcode':'int'})

# setting the country as Australia
nomi = pgeocode.Nominatim('au')

# get lat/lng values for each postcode value and add to the same dataframe
df['lat'] = df['Postcode'].map(lambda Postcode: nomi.query_postal_code(Postcode).latitude)
df['lng'] = df['Postcode'].map(lambda Postcode: nomi.query_postal_code(Postcode).longitude)
print(df.head())


# In[16]:


# removing any NAN and inf values from the dataframe
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
print(len(df))


# In[17]:


#features = df[['MentalHealthCostPerClaimant','addictionCostPerClaimant','obesityCostPerClaimant','musculoskeletalCostPerClaimant']]
features = df[['MentalHealthCostPerClaimant','addictionCostPerClaimant']]
print(features.head())


# In[18]:


# elbow method to find number of clusters

Sum_of_squared_distances = []
K = range(2, 10)

for k in K:
    km = KMeans(n_clusters=k, max_iter=200, n_init=10)
    km = km.fit(features)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[19]:


# create kmeans model/object
kmeans = KMeans(
    init="random",
    n_clusters=5,
    n_init=10,
    max_iter=300,
    random_state=42
)

# do clustering
kmeans.fit(features)
# save results
labels = kmeans.labels_


# In[20]:


# send back into dataframe and display it
df['cluster'] = labels
# display the number of mamber each clustering
_clusters = df.groupby('cluster').count()
print(_clusters)


# In[21]:


def get_center_latlong(df):
    # get the center of my map for plotting
    centerlat = (df['lat'].max() + df['lat'].min()) / 2
    centerlong = (df['lng'].max() + df['lng'].min()) / 2
    return centerlat, centerlong


# In[22]:


# colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', \
#      'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', \
#      'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray', \
#      'black', 'lightgray', 'red', 'blue', 'green', 'purple', \
#      'orange', 'darkred', 'lightred', 'beige', 'darkblue', \
#      'darkgreen', 'cadetblue', 'darkpurple','pink', 'lightblue', \
#      'lightgreen', 'gray', 'black', 'lightgray' ]

colors = ['red', 'blue', 'green', 'orange', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray' ]

center = get_center_latlong(df)
print(center)
m = folium.Map(location=center, zoom_start=4)


# In[23]:


for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=5,
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])]
    ).add_to(m)


# In[24]:


m


# In[25]:


df.head()


# In[26]:


import findspark
findspark.init()
findspark.find()


# In[27]:


from pyspark.sql import SparkSession
# Create SparkSession
spark = SparkSession.builder.appName('SparkByExample').getOrCreate()


# In[28]:


sparkDF=spark.createDataFrame(df) 
sparkDF.printSchema()
sparkDF.show()


# In[29]:


sparkDF.createOrReplaceTempView("COSTS")

sqlDF = spark.sql("SELECT * FROM COSTS")
sqlDF.show()


# In[30]:


pandasDF = sqlDF.toPandas()
print(pandasDF)


# In[31]:


original_df = df
df.head()


# In[32]:


center = get_center_latlong(df)
m0 = folium.Map(location=center, zoom_start=4)

cluster0_df = df[df['cluster']==0]
cluster0_df.head()

for _, row in cluster0_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=5,
        fill=True,
        color='red',
        fill_color='red',
    ).add_to(m0)
m0


# In[33]:


center = get_center_latlong(df)
m1 = folium.Map(location=center, zoom_start=4)

cluster1_df = df[df['cluster']==1]
cluster1_df.head()

for _, row in cluster1_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=5,
        fill=True,
        color='blue',
        fill_color='blue',
    ).add_to(m1)
m1


# In[34]:


center = get_center_latlong(df)
m2 = folium.Map(location=center, zoom_start=4)

cluster2_df = df[df['cluster']==2]
cluster2_df.head()

for _, row in cluster2_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=5,
        fill=True,
        color='green',
        fill_color='green',
    ).add_to(m2)
m2


# In[35]:


center = get_center_latlong(df)
m3 = folium.Map(location=center, zoom_start=4)

cluster3_df = df[df['cluster']==3]
cluster3_df.head()

for _, row in cluster3_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=5,
        fill=True,
        color='yellow',
        fill_color='yellow',
    ).add_to(m3)
m3


# In[36]:


center = get_center_latlong(df)
m4 = folium.Map(location=center, zoom_start=4)

cluster4_df = df[df['cluster']==4]
cluster4_df.head()

for _, row in cluster4_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=5,
        fill=True,
        color='blue',
        fill_color='blue',
    ).add_to(m4)
m4


# In[37]:


center = get_center_latlong(df)
m5 = folium.Map(location=center, zoom_start=4)

cluster5_df = df[df['cluster']==5]
cluster5_df.head()

for _, row in cluster5_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=5,
        fill=True,
        color='blue',
        fill_color='blue',
    ).add_to(m5)
m5

