#!/usr/bin/env python
# coding: utf-8

# # AI Lab-4
# ## Data reading, analysis and visualization
# ### Author : Abdul Basit

# #### Lab Tasks

# #### Take an image of a “Car” and a “Cup”. Perform following tasks:
# > Read both images
# 
# > Resize image to 256 by 256
# 
# > Show both images
# 
# > Convert images to gray scale
# 
# > Normalize both images
# 
# > Show grayscale images
# 
# > Find contrast, energy and mean of both images
# 
# > Plot contrast, energy and mean of both images
# 
# > Discuss which feature among contrast, energy and is best for classification
# 
# > Perform the edge detection on both images and show its resultant image

# In[1]:


#REading both images
import cv2 
import numpy as np
import matplotlib.pyplot as plt
car = cv2.imread('../input/car-cup/download.jpg')
cup = cv2.imread('../input/car-cup/cup.jpg')
print(plt.imshow(car))


# In[2]:


#reading second image
print(plt.imshow(cup))


# ### Exercise# 2 and 3

# In[3]:


# Resizing both image to 256 by 256
car = cv2.resize(car,(256,256))
print(car.shape)

#showing both images
print('Image 1')
plt.imshow(car)


# In[4]:


cup = cv2.resize(cup,(256,256))
print(cup.shape)
print('Image 2')
plt.imshow(cup)


# ### Exercise# 4 and 6

# In[5]:


#Converting both images to gray scale
gray_car = cv2.cvtColor(car,cv2.COLOR_BGR2GRAY)
gray_cup = cv2.cvtColor(cup,cv2.COLOR_BGR2GRAY)

plt.imshow(gray_car)


# In[6]:


plt.imshow(gray_cup)


# ### Exercise# 5

# In[7]:


#Normalizing both images
normalize = np.zeros((250,250))
norm_car = cv2.normalize(gray_car,normalize,0,255,cv2.NORM_MINMAX)
norm_cup = cv2.normalize(gray_cup,normalize,0,255,cv2.NORM_MINMAX)

#image 1 normalized
plt.imshow(norm_car)


# In[8]:


#Image 2 normailized
plt.imshow(norm_cup)


# ### Exercise# 7 and 8

# In[9]:


# Calculating contrast, energy and mean of both images
# for the applying some techniques to find

min_car = np.min(gray_car)
min_cup = np.min(gray_cup)

max_car = np.max(gray_car)
max_cup = np.max(gray_cup)

# formulla for the contrast is as following

cont_car = (max_car-min_car)/(max_car+min_car)
cont_cup = (max_cup-min_cup)/(max_cup+min_cup)

print('Contrast of car is : ',cont_car)
print('Contrast of cup is : ',cont_cup)
print(plt.bar(['car','cup'],[cont_car,cont_cup]))


# In[10]:


#calculating standrad deviation of the image for energy
car_std = np.std(car)
cup_std = np.std(cup)

print(car_std,cup_std)
print(print(plt.bar(['car','cup'],[car_std,cup_std])))


# In[11]:


#Calculating mean of both images

mean1= car.mean()
mean2 = cup.mean()

print(mean1,mean2)
print(print(plt.bar(['car','cup'],[mean1,mean2])))


# ### Exercise# 9

# In[12]:


energy_dif = (car_std) - (cup_std)
cont_dif = (cont_car) - (cont_cup)
mean_dif = (mean1) - (mean2)

print(energy_dif)
print(cont_dif)
print(mean_dif)

In this case Mean of the images gives highest difference, so we can consider mean feature for the classification.
# # Exercise# 10

# In[13]:


#Edge detectionn in python
car_edg = cv2.Canny(gray_car, threshold1=30, threshold2=100)
plt.imshow(car_edg)


# In[14]:


cup_edg = cv2.Canny(gray_cup, threshold1=30, threshold2=100)
plt.imshow(cup_edg)

