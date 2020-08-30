#!/usr/bin/env python
# coding: utf-8

# ## To Explore Unsupervised Machine Learning

#  Importing all the libraries

# In[68]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans


# Reading The Data

# In[69]:


data = pd.read_csv('Iris.csv')


# In[70]:


data.head(10)


# Data Exploration

# In[71]:


data.info


# In[72]:


data.shape


# In[73]:


data.corr()


# In[74]:


data.describe()


# In[75]:


x = data.iloc[: , :-1].values  
y = data.iloc[: , -1].values  
print(x)
print(y)


# Visualizing The Data

# In[76]:


sns.pairplot(data)


# In[77]:


x = data["SepalLengthCm"]
y = data["SepalWidthCm"]
sns.scatterplot(x,y,color='red')
plt.title("SepalWidthCm vs SepalLengthCm")


# In[78]:


x = data["PetalLengthCm"]
y = data["PetalWidthCm"]
sns.scatterplot(x,y,color='blue')
plt.title("PetalLengthCm vs PetalWidthCm")


# Finding The Optimum Number of Clusters for k-means Classification

# In[84]:


x = data.iloc[:, [0, 1, 2, 3]].values

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 5)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Optimun number of cluster is 3

# Applying kmeans to The Dataset

# In[85]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 5)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)


# Visualising The Clusters

# In[86]:


plt.figure(figsize = (8,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 30, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 30, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],s = 30, c = 'green', label = 'Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 30, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:





# In[ ]:




