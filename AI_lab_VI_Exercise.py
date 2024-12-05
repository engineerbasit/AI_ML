#!/usr/bin/env python
# coding: utf-8

# # AI Lab-6
# ## Regression, feature selection and PCA
# ### Author : Abdul Basit

# #### Lab Tasks

# > 1. Load and define briefly the diabetes dataset available in sklearn.
# > 2. Write brief summary about Diabetes Dataset
# > 3. Print, Check and define keys and shape of diabetes dataset.
# > 4. Split the dataset into train and test.
# > 5. Fit the model on test data and compute accuracy using Linear Regression.
# > 6. Make slots of features on data set and evaluate using regression metrics.
# > 7. Apply PCA and analyze the results with results without PCA.
# > 8. Apply permutation importance function and plot feature importances.

# In[2]:


# importing all necessary libraries
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn as sk


# ## Brief Description of the diabetes dataset
# 
# This dataset contains information of diabetes patients information containing different features. We will use this dataset to predict whether a patient has diabetes  or not by training a model using this dataset.

# ## Importing diabetes data set

# In[3]:


dib = datasets.load_diabetes()


# Keys of the dataset

# In[3]:


a = dib.keys()
a


# In[4]:


d_data = dib.data
# storing data in a variable


# In[5]:


dib.data.shape


# In[6]:


# printing all the keys availible in the dataset
for i in a:
    print (dib[i])
    print('\n')


# # Separating the features and labels

# In[7]:


features = dib.data
labels = dib.target
features.shape, labels.shape


# ## Converting data from arrays to give it a better look:
# 

# In[8]:


df_dib = pd.DataFrame(features, columns = dib.feature_names)
df_dib['Values'] = dib.target
df_dib.head()


# In[9]:


# X is features and Y is labels
X = df_dib.loc[:,df_dib.columns!='Values']
Y = df_dib.Values

X,Y


# ## Spliting and data for taining the model

# In[10]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size = 0.25 , shuffle = True)
Xtrain.shape, Ytest.shape, Xtest.shape, Ytest.shape


# # Fitting the model using linear regression 

# In[11]:


from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()


# In[12]:


LinReg.fit(Xtrain,Ytrain)


# In[13]:


Acc = LinReg.score(Xtest,Ytest)
Acc


# # Selecting a feature from the dataset:

# In[14]:


dXtest = pd.DataFrame(Xtest,columns = dib.feature_names)
dXtrain = pd.DataFrame(Xtrain, columns = dib.feature_names)
dXtest = dXtest[['age']]
dXtrain = dXtrain[['sex']]


# In[15]:


LinReg.fit(dXtrain,Ytrain)

plt.xlabel('Features Value')
plt.ylabel('Price of House')
plt.title('Linear Regression Visulization on Single Feature')
plt.scatter(dXtest, Ytest, color = 'black', label = 'Origional Data Distribution')
plt.plot(dXtest, LinReg.predict(dXtest),color= 'blue',linewidth = 3, label = 'Linear Regression Line trying to fit on data')
plt.legend(loc = 'upper right')


# ##  Making slots of features on data set and evaluating slots using regression metrics

# In[16]:


s1xtr = Xtrain[['age','sex','bmi']]
s2xtr = Xtrain[['s1','s2','s3']]
s3xtr = Xtrain[['s3','s4','s6']]
s4xtr = Xtrain[['bp','s5','s6','age']]

s1xts = Xtest[['age','sex','bmi']]
s2xts = Xtest[['s1','s2','s3']]
s3xts = Xtest[['s3','s4','s6']]
s4xts = Xtest[['bp','s5','s6','age']]


# # Fitting and evaluating on all slot:
# 

# In[17]:


LinReg.fit(s1xtr,Ytrain)
LinReg.score(s1xts,Ytest)


# In[18]:


LinReg.fit(s2xtr,Ytrain)
LinReg.score(s2xts,Ytest)


# In[19]:


LinReg.fit(s3xtr,Ytrain)
LinReg.score(s3xts,Ytest)


# In[20]:


LinReg.fit(s4xtr,Ytrain)
LinReg.score(s4xts,Ytest)


# # Calculating all metrics for all slots using linear regression

# In[21]:


from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error
LinReg.fit(s1xtr, Ytrain)
print('Mean Absolute Error of slot 1:',mean_absolute_error(Ytest,LinReg.predict(s1xts)))
print('Mean Squared Error of slot 1:',mean_squared_error(Ytest,LinReg.predict(s1xts)))
print('Explained Variance Score of slot 1:',explained_variance_score(Ytest,LinReg.predict(s1xts)))


# In[22]:


LinReg.fit(s2xtr, Ytrain)
print('Mean Absolute Error of slot 2:',mean_absolute_error(Ytest,LinReg.predict(s2xts)))
print('Mean Squared Error of slot 2:',mean_squared_error(Ytest,LinReg.predict(s2xts)))
print('Explained Variance Score of slot 2:',explained_variance_score(Ytest,LinReg.predict(s2xts)))


# In[23]:


LinReg.fit(s3xtr, Ytrain)
print('Mean Absolute Error of slot 3:',mean_absolute_error(Ytest,LinReg.predict(s3xts)))
print('Mean Squared Error of slot 3:',mean_squared_error(Ytest,LinReg.predict(s3xts)))
print('Explained Variance Score of slot 3:',explained_variance_score(Ytest,LinReg.predict(s3xts)))


# In[24]:


LinReg.fit(s4xtr, Ytrain)
print('Mean Absolute Error of slot 4:',mean_absolute_error(Ytest,LinReg.predict(s4xts)))
print('Mean Squared Error of slot 4:',mean_squared_error(Ytest,LinReg.predict(s4xts)))
print('Explained Variance Score of slot 4:',explained_variance_score(Ytest,LinReg.predict(s4xts)))


# ## Calculating all metrics for test dataset containing all features

# In[25]:


LinReg.fit(Xtrain, Ytrain)
print('Mean Absolute Error:',mean_absolute_error(Ytest,LinReg.predict(Xtest)))
print('Mean Squared Error:',mean_squared_error(Ytest,LinReg.predict(Xtest)))
print('Explained Variance Score:',explained_variance_score(Ytest,LinReg.predict(Xtest)))


# # Principal Components Analysis (PCA)

# In[26]:


# Model Accuracy is 
Acc


# In[27]:


pca = PCA(n_components = 6)
pca.fit(d_data)
pca.explained_variance_ratio_


# In[28]:


pca = PCA(n_components = 9)
trX = pca.fit_transform(Xtrain)
tsX = pca.transform(Xtest)
trX


# In[29]:


#printing shape of the transformed data
trX.shape, tsX.shape


# In[30]:


LinReg.fit(X,Y)
X.shape,Y.shape


# ## Inspecting the feature importance using sklearn
# 

# In[31]:


from sklearn.inspection import permutation_importance
results = permutation_importance(LinReg,X,Y,n_repeats = 10, random_state = 0)
for i in results.importances_mean.argsort():
    print('Features', X.columns[i], ' : " " has importance ', results.importances_mean[i])


# In[32]:


feature = []
importances = []

for i in results.importances_mean.argsort():
    feature.append(i)
    importances.append(results.importances_mean[i])
    
    
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Plot of importances of corrosponding feature')
plt.bar(range(len(importances)),importances, align = 'center')
plt.xticks(range(len(feature)),X.columns[feature],rotation = 1)

