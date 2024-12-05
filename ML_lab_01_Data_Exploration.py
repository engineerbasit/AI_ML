#!/usr/bin/env python
# coding: utf-8

# ### ML Lab-01
# # Data Exploration
# ## Abdul Basit

# #### Task_01

# In[30]:


#Important Libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
# Seaborn visualization library
import seaborn as sns


# In[31]:


# loading breast cancer dataset from sklearn
bcdata = load_breast_cancer()


# In[32]:


bcdata.keys()


# In[33]:


#list of class labels

list(bcdata.target_names)


# In[34]:


len(bcdata)


# In[35]:


print(bcdata.DESCR)


# In[36]:


#Creating dataframe from the data
df = pd.DataFrame(bcdata.data, columns = bcdata.feature_names)


# In[37]:


df.shape


# In[38]:


#observations in the data
len(df)


# In[39]:


#features of data
bcdata.feature_names


# In[40]:


df.head()


# In[41]:


df.tail()


# In[42]:


x = df['mean radius']
y = df['mean area']

plt.scatter(x, y)
plt.xlabel('mean radius')
plt.ylabel('mean area')
plt.show()


# In[43]:


df.plot.scatter(x='mean radius', y = 'mean area')


# In[44]:


df.plot()


# In[45]:


df[['mean radius','mean area']].plot()


# In[46]:


df['mean radius'].plot()


# In[47]:


df['mean area'].plot()


# In[48]:


corl = df.corr(method ='pearson') 
corl


# In[49]:


df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)


# In[50]:


sns.heatmap(corl, annot = True)

plt.show()


# In[51]:


sns.heatmap(corl)

plt.show()


# #### Lab task_2

# In[52]:


pd.set_option('display.max_columns', None)


# In[53]:


df2 = pd.read_csv("USA_cars_datasets.csv")


# In[54]:


df2.keys()


# In[56]:


# Examples
len(df2)


# In[58]:


import numpy as npfrom 


# In[61]:


from sklearn import datasets

