#!/usr/bin/env python
# coding: utf-8

# # AI Lab-4
# ## Data reading, analysis and visualization
# ### Author : Abdul Basit

# ### Lab Practice
# 

# In[58]:


#Importing the data from sklearn:
from sklearn import datasets
iris = datasets.load_iris()
iris.data.shape


# In[59]:


import pandas as pd
#creating a dataframe

print(iris.data[0:5])
print(iris.target)
print(iris.feature_names)


# In[60]:


data = pd.DataFrame(iris.data,columns = iris.feature_names)
print(data.head())


# In[61]:


data.to_csv('File.csv',index = False)
pd.read_csv('./File.csv')


# In[62]:


x = pd.read_csv('./File.csv',delimiter = ',')
x['Target'] = iris.target
x.tail()


# In[63]:


len(x)


# In[64]:


#Plotting data 
x.plot()


# In[65]:


#To plot any specified columns from the data
x[['sepal length (cm)','sepal width (cm)', 'petal length (cm)',]].plot()


# In[66]:


#Plotting individual column from the iris data
x[['petal width (cm)','sepal length (cm)']].plot()


# In[67]:


#Finding which information is present in the data
iris.keys()


# In[78]:


# To print any individual column from the iris data.
iris.target


# In[69]:


x['petal width (cm)'].plot()


# In[84]:


x.drop(['Target'],axis=1).plot(xlabel = 'Data Index', ylabel = 'Size of the corrosponding features')


# In[86]:


#Scatter plot of the data
x.plot.scatter(x = 'sepal length (cm)', y = 'sepal width (cm)')


# In[89]:


#plotting the scatter with colours
x.plot.scatter(x = 'sepal length (cm)', y = 'sepal width (cm)', c = iris.target,colormap = 'viridis')


# In[90]:


x[x['Target']==0].mean().drop(['Target']).plot.bar(title = iris.target_names[0]+'Features Mean')


# In[91]:


x[x['Target']==1].mean().drop(['Target']).plot.bar(title = iris.target_names[0]+'Features Mean')


# In[92]:


x[x['Target']==2].mean().drop(['Target']).plot.bar(title = iris.target_names[0]+'Features Mean')


# ### Loading Mnist dataset containing digit image and checking the shape

# In[94]:


from sklearn import datasets
# dgi = digit_images

# loading dataset digits
dgi = datasets.load_digits()

#displying the shape of image
dgi.data.shape


# In[95]:


#displying data 2 of the image
dgi.data[2]


# In[100]:


import matplotlib.pyplot as plt
image = dgi.data[8].reshape(8,8)

plt.imshow(image, cmap = 'gray')


# # Introduction to CV2

# In[2]:


# Dataset Link
# https://www.kaggle.com/puneet6060/intel-image-classification


# In[101]:


import cv2
path = '../input/intel-image-classification/seg_train/seg_train/buildings/1012.jpg'
image = cv2.imread(path)
plt.imshow(image)


# In[116]:


# Changing image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray)


# In[117]:


#Printing shape of Gray scale and color image
image.shape,image_gray.shape


# In[115]:


normalized_image = image/255
normalized_image


# In[111]:


#image = cv2.resize(image,(100,100))
#image.shape, plt.imshow(image)


# In[118]:


print("Origional Shape : ", image.shape)


# In[121]:


#Reshaping the image into a vector
print("Origional Shape : ", image.shape)
vector = image.reshape(30000)
print("Shape Reshaped image in to vector : ", vector.shape)
print("Total values in vector : ", len(vector))
vector


# In[124]:


print("Previous Shape : ", vector.shape)
return_image = vector.reshape(100,100,3)
print("New Shape as an image : ",return_image.shape)
plt.imshow(return_image)


# In[127]:


import numpy as np
import glob
import cv2

# path = p
p1 = glob.glob('../input/intel-image-classification/seg_train/seg_train/buildings/*.jpg')
cv_img = []
for img in p1:
    n = cv2.imread(img)
    n = cv2.resize(n,(200,200))
arr1 = np.asarray(cv_img)
print(arr1.shape)


# In[ ]:




