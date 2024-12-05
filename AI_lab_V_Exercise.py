#!/usr/bin/env python
# coding: utf-8

# # AI Lab-5
# ## Supervised Machine Learning-Sklearn
# ### Author : Abdul Basit

# #### Lab Tasks

# > Study and Explore Mnist or Digits dataset available in the sklearn.
# 
# > Print the keys available in the data set.
# 
# > Define all the keys briefly to define the dataset.
# 
# > Check and define the shape of dataset.
# 
# > Separate the features and split the dataset into train/test and shuffle it.
# 
# > Fit the model using KNN classifier by selecting appropriate value of K you think.
# 
# > Compute accuracy of the model on the test data.
# 
# > Plot confusion matrix of the data and its classification report.
# 
# > Evaluate the model in terms of overfitting and underfitting.

# # Loading dataset of minst Digits dataset

# In[1]:


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
mnist = datasets.load_digits()
mnist.target_names
A= mnist.data
B = mnist.target
A,B


# In[2]:


sample = 1000

image = A[sample]
image = image.reshape(8,8)

plt.imshow(image)
plt.show()


# # Printing Keys of the dataset

# In[3]:


mnist.keys()


# # # Defining keys in the dataset
Data = Contans data

Target = labels of the data

Frame  = Contains frames

Feature_names = All frame names

target_name = Target label names

images = Contains Images of dataset

DESCR = This key contains Description of the data
# # Checking shape of the dataset

# In[4]:


mnist.data.shape


# The shape of the dataset is 1797x64 

# # Seperating features of dataset

# In[5]:


features = mnist.feature_names
features


# In[6]:


# A is data in dataset
# B is target in the dataset
df = pd.DataFrame(A, columns = mnist.feature_names)

df['target'] = B
df.head()


# # Spliting and Shuffling features for training and testing

# In[7]:


from sklearn.model_selection import train_test_split
Atrain, Atest, Btrain, Btest = train_test_split(A,B,test_size = 0.2, shuffle = True)

Atrain.shape, Atest.shape, Btrain.shape, Btest.shape


# # Fitting the model using KNN classifer

# In[8]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=15)
model.fit(Atrain,Btrain)


# In[9]:


results = model.predict(Atest)
results


# # Computing Accuracy of the model

# In[10]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(results,Btest)
accuracy


# # Confusion matrix of the model

# In[11]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model,Atest,Btest)


# # Classification of the model

# In[12]:


from sklearn.metrics import classification_report
print(classification_report(Btest,results))


# In[13]:


total_train_accuracy = []
total_test_accuracy = []
neighbors = np.arange(1,9)


# In[14]:


for i in range(8):
    knn= KNeighborsClassifier(n_neighbors = i+1)
    #fitting the classifier to training data
    knn.fit(Atrain,Btrain)
    
    train_accuracy = knn.score(Atrain,Btrain)
    
    test_accuracy = knn.score(Atest,Btest)
    
    total_train_accuracy.append(train_accuracy)
    total_test_accuracy.append(test_accuracy)


# In[15]:


#Generating a plot of the tained and tested data
plt.title('KNN: Verifying Number of Neighbours')
plt.plot(neighbors,total_train_accuracy, label = 'Train Accuracy' )
plt.plot(neighbors,total_test_accuracy, label = 'Test Accuracy' )
plt.legend()
plt.xlabel('Numbers of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# The model is underfitting because as the training data is increasing the accuracy is decreasing
