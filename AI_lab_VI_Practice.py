#!/usr/bin/env python
# coding: utf-8

# # AI Lab-6
# ## Regression, feature selection and PCA
# ### Author : Abdul Basit

# ### Lab Practice

# In[2]:


# importing all necessary libraries
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sklearn as sk


# In[3]:


boston = datasets.load_boston()


# In[3]:


a = boston.keys()
a


# In[4]:


boston.data.shape


# In[5]:


for i in a:
    print (boston[i])
    print('\n')


# In[6]:


boston.data.shape


# Separating the features and labels

# In[7]:


b_data = boston.data
b_target = boston.target
b_data.shape, b_target.shape


# Converting data from arrays to give it a better look:
# 

# In[8]:


df_boston = pd.DataFrame(b_data, columns = boston.feature_names)
df_boston['Price'] = boston.target
df_boston.head()


# In[9]:


boston.keys()


# In[10]:


A = df_boston.loc[:,df_boston.columns!='Price']
B = df_boston.Price
A.shape, B.shape


# In[11]:


from sklearn.model_selection import train_test_split
Atrain, Atest, Btrain, Btest = train_test_split(A,B,test_size = 0.25 , shuffle = True)
Atrain.shape, Btest.shape, Atest.shape, Btest.shape


# # Linear Regression:

# In[12]:


from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()


# In[13]:


LinReg.fit(Atrain,Btrain)


# In[14]:


Acc = LinReg.score(Atest,Btest)
Acc


# # Selecting a feature from the dataset:

# In[15]:


dAtest = pd.DataFrame(Atest,columns = boston.feature_names)
dAtrain = pd.DataFrame(Atrain, columns = boston.feature_names)
dAtest = dAtest[['LSTAT']]
dAtrain = dAtrain[['LSTAT']]


# Plotting the selected feature value distribution along with price of the house and model line:

# In[16]:


LinReg.fit(dAtrain,Btrain)

plt.xlabel('Features Value')
plt.ylabel('Price of House')
plt.title('Linear Regression Visulization on Single Feature')
plt.scatter(dAtest, Btest, color = 'black', label = 'Origional Data Distribution')
plt.plot(dAtest, LinReg.predict(dAtest),color= 'blue',linewidth = 3, label = 'Linear Regression Line trying to fit on data')
plt.legend(loc = 'upper right')


# In[17]:


s1Atr = Atrain[['CRIM','ZN','INDUS']]
s2Atr = Atrain[['CHAS','NOX','RM']]
s3Atr = Atrain[['AGE','DIS','RAD']]
s4Atr = Atrain[['TAX','PTRATIO','B','LSTAT']]

s1Ats = Atest[['CRIM','ZN','INDUS']]
s2Ats = Atest[['CHAS','NOX','RM']]
s3Ats = Atest[['AGE','DIS','RAD']]
s4Ats = Atest[['TAX','PTRATIO','B','LSTAT']]


# # Fitting and evaluating on all slot:
# 

# In[18]:


LinReg.fit(s1Atr,Btrain)
LinReg.score(s1Ats,Btest)


# In[19]:


LinReg.fit(s2Atr,Btrain)
LinReg.score(s2Ats,Btest)


# In[20]:


LinReg.fit(s3Atr,Btrain)
LinReg.score(s3Ats,Btest)


# In[21]:


LinReg.fit(s4Atr,Btrain)
LinReg.score(s4Ats,Btest)


# # Calculating all metrics for all slots using linear regression

# In[22]:


from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error
LinReg.fit(s1Atr, Btrain)
print('Mean Absolute Error of slot 1:',mean_absolute_error(Btest,LinReg.predict(s1Ats)))
print('Mean Squared Error of slot 1:',mean_squared_error(Btest,LinReg.predict(s1Ats)))
print('Explained Variance Score of slot 1:',explained_variance_score(Btest,LinReg.predict(s1Ats)))


# In[23]:


LinReg.fit(s2Atr, Btrain)
print('Mean Absolute Error of slot 2:',mean_absolute_error(Btest,LinReg.predict(s2Ats)))
print('Mean Squared Error of slot 2:',mean_squared_error(Btest,LinReg.predict(s2Ats)))
print('Explained Variance Score of slot 2:',explained_variance_score(Btest,LinReg.predict(s2Ats)))


# In[24]:


LinReg.fit(s3Atr, Btrain)
print('Mean Absolute Error of slot 3:',mean_absolute_error(Btest,LinReg.predict(s3Ats)))
print('Mean Squared Error of slot 3:',mean_squared_error(Btest,LinReg.predict(s3Ats)))
print('Explained Variance Score of slot 3:',explained_variance_score(Btest,LinReg.predict(s3Ats)))


# In[25]:


LinReg.fit(s4Atr, Btrain)
print('Mean Absolute Error of slot 4:',mean_absolute_error(Btest,LinReg.predict(s4Ats)))
print('Mean Squared Error of slot 4:',mean_squared_error(Btest,LinReg.predict(s4Ats)))
print('Explained Variance Score of slot 4:',explained_variance_score(Btest,LinReg.predict(s4Ats)))


# Calculating all metrics for test dataset containing all features

# In[26]:


LinReg.fit(Atrain, Btrain)
print('Mean Absolute Error:',mean_absolute_error(Btest,LinReg.predict(Atest)))
print('Mean Squared Error:',mean_squared_error(Btest,LinReg.predict(Atest)))
print('Explained Variance Score:',explained_variance_score(Btest,LinReg.predict(Atest)))


# # Principal Components Analysis (PCA)

# In[27]:


# Model Accuracy is 
Acc


# In[28]:


pca = PCA(n_components = 6)
pca.fit(b_data)
pca.explained_variance_ratio_


# In[29]:


pca = PCA(n_components = 9)
trA = pca.fit_transform(Atrain)
tsA = pca.transform(Atest)
trA


# In[30]:


#printing shape of the transformed data
trA.shape, tsA.shape


# In[31]:


LinReg.fit(A,B)
A.shape,B.shape


# Inspecting the feature importance using sklearn
# 

# In[32]:


from sklearn.inspection import permutation_importance
results = permutation_importance(LinReg,A,B,n_repeats = 10, random_state = 0)
for i in results.importances_mean.argsort():
    print('Features', A.columns[i], ' : " " has importance ', results.importances_mean[i])


# In[ ]:




