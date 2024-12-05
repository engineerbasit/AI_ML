#!/usr/bin/env python
# coding: utf-8

# # AI Lab-5
# ## Supervised Machine Learning-Sklearn
# ### Author : Abdul Basit

# ### Lab Practice
# 

# In[1]:


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = datasets.load_iris()
type(iris)


# In[2]:


iris.keys()


# In[3]:


iris.feature_names


# In[5]:


iris.data.shape


# In[6]:


iris.target.shape


# In[7]:


X = iris.data
Y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)

df['target'] = Y
df.head()


# **KNN Classifier**

# In[8]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=6)


# In[9]:


model.fit(X,Y)


# In[10]:


sample = np.array([[5,3.4,1.3,0.2],[4.8,3.1,1.4,0.2],[5.9,3.0,5.1,1.8]])
result = model.predict(sample)
result


# In[11]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size = 0.2, shuffle = True)

Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape


# In[12]:


model.fit(Xtrain,Ytrain)


# In[13]:


results = model.predict(Xtest)
results


# In[14]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(results,Ytest)
accuracy


# In[15]:


#Alternate direct method to calculate the accuracy:
model.score(Xtest,Ytest)


# In[16]:


#Plotting the confusion matrix of the model on test: 
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model,Xtest,Ytest)


# In[17]:


from sklearn.metrics import classification_report
print(classification_report(Ytest,results))


# # Examining Under fitting and over fitting and in Machine learning: 

# In[18]:


total_train_accuracy = []
total_test_accuracy = []
neighbors = np.arange(1,9)


# Using for loop to iterate over each value of K from 1 to 8 to calculate and store the values of training and testing accuracies in defined list:

# In[19]:


for i in range(8):
    knn= KNeighborsClassifier(n_neighbors = i+1)
    #fitting the classifier to training data
    knn.fit(Xtrain,Ytrain)
    
    train_accuracy = knn.score(Xtrain,Ytrain)
    
    test_accuracy = knn.score(Xtest,Ytest)
    
    total_train_accuracy.append(train_accuracy)
    total_test_accuracy.append(test_accuracy)
    


# In[20]:


#Generating a plot of the tained and tested data
plt.title('KNN: Verifying Number of Neighbours')
plt.plot(neighbors,total_train_accuracy, label = 'Train Accuracy' )
plt.plot(neighbors,total_test_accuracy, label = 'Test Accuracy' )
plt.legend()
plt.xlabel('Numbers of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# # Plotting an image of from the Mnist dataset:  
# # Loading the Mnist dataset from sklearn and separating the features and labels: 

# In[21]:


mnist = datasets.load_digits()
mnist.target_names
A= mnist.data
B = mnist.target
A,B


# In[22]:


sample = 9
image = A[sample]
image = image.reshape(8,8)

plt.imshow(image)
plt.show()


# In[ ]:




