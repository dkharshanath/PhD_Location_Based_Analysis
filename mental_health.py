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
df = pd.read_csv('mental_health_year_postcode_cost.csv')
print(len(df))


# In[3]:


# removing any NAN and inf values from the dataframe
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
print(len(df))


# In[4]:


df.head()


# In[5]:


original_df = df
original_df.head()


# In[6]:


#df = df[(df['FinancialYear'] == 2017)]
#df.head()


# In[7]:


#df = df.sample(n=1000,random_state=1)


# In[ ]:


# convert postcode column to integer data type
df = df.astype({'Postcode':'int'})

# setting the country as Australia
nomi = pgeocode.Nominatim('au')

# get lat/lng values for each postcode value and add to the same dataframe
df['lat'] = df['Postcode'].map(lambda Postcode: nomi.query_postal_code(Postcode).latitude)
df['lng'] = df['Postcode'].map(lambda Postcode: nomi.query_postal_code(Postcode).longitude)
print(df.head())


# In[ ]:


features = df[['lat', 'lng']]
print(features.head())


# In[ ]:


# removing any NAN and inf values from the dataframe
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
print(len(df))


# In[ ]:


features = df[['lat', 'lng']]
print(features.head())


# In[ ]:


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


# In[ ]:


# create kmeans model/object
kmeans = KMeans(
    init="random",
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)

# do clustering
kmeans.fit(features)
# save results
labels = kmeans.labels_


# In[ ]:


# send back into dataframe and display it
df['cluster'] = labels
# display the number of mamber each clustering
_clusters = df.groupby('cluster').count()
print(_clusters)


# In[ ]:


colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',      'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',      'darkpurple', 'pink', 'lightblue', 'lightgreen', 'gray',      'black', 'lightgray', 'red', 'blue', 'green', 'purple',      'orange', 'darkred', 'lightred', 'beige', 'darkblue',      'darkgreen', 'cadetblue', 'darkpurple','pink', 'lightblue',      'lightgreen', 'gray', 'black', 'lightgray' ]

lat = df.iloc[0]['lat']
lng = df.iloc[0]['lng']

map = folium.Map(location=[lng, lat], zoom_start=2)


# In[ ]:


for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=12, 
        weight=2, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])]
    ).add_to(map)


# In[ ]:


map


# In[ ]:


df.head()


# In[ ]:


import findspark
findspark.init()
findspark.find()


# In[ ]:


from pyspark.sql import SparkSession
#Create SparkSession
spark = SparkSession.builder.appName('SparkByExample').getOrCreate()


# In[ ]:


sparkDF=spark.createDataFrame(df) 
sparkDF.printSchema()
sparkDF.show()


# In[ ]:


sparkDF.createOrReplaceTempView("COSTS")

sqlDF = spark.sql("SELECT * FROM COSTS")
sqlDF.show()


# In[ ]:


query = """
SELECT cluster, COUNT(DISTINCT PersonID) AS ClaimantsCount, SUM(MentalHealthCost) AS TotalCostPerCluster,
SUM(MentalHealthCost)/COUNT(DISTINCT PersonID) AS CostPerClaimant
FROM COSTS
GROUP BY Cluster
ORDER BY Cluster
"""

sqlDF = spark.sql(query)
sqlDF.show()


# In[ ]:


query = """
SELECT FinancialYear, Cluster, COUNT(DISTINCT PersonID) AS ClaimantsCount, SUM(MentalHealthCost) AS TotalCostPerCluster,
SUM(MentalHealthCost)/COUNT(DISTINCT PersonID) AS CostPerClaimant
FROM COSTS
GROUP BY FinancialYear, Cluster
ORDER BY FinancialYear, Cluster
"""

sqlDF = spark.sql(query)
sqlDF.show()


# In[ ]:


pandasDF = sqlDF.toPandas()
print(pandasDF)


# In[ ]:


plot_me = pandasDF.drop(['ClaimantsCount', 'TotalCostPerCluster'], axis=1)
plot_me.head()


# In[ ]:


import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(data=plot_me, x='FinancialYear', y='CostPerClaimant', hue='Cluster')

