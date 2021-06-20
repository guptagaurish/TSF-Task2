#!/usr/bin/env python
# coding: utf-8

# # NAME - GAURISH GUPTA

# # Task 2-K Means Clustering

# ### From the given ‘Iris’ dataset,we have predict the optimum number of clusters and represent it visually

# ### IMPORTS

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the dataset

# In[3]:


import seaborn as sns
iris = sns.load_dataset('iris')


# In[4]:


iris.head()


# # Finding the optimum number of clusters for k-means classification

# In[5]:


x = iris.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 101)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# # Plotting the results into one line Graph

# In[8]:


sns.set_style("whitegrid")
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# #### From above graph using elbow method we can find the optimum number of clusters.This is when the elbow occurs when within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration. Therefore,choosing number of clusters as "3"

# In[9]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 101)
y_kmeans = kmeans.fit_predict(x)


# ## Visualizing the Clusters

# In[10]:


plt.figure(figsize=(10,6))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# ### Task Completed Successfully

# In[ ]:




