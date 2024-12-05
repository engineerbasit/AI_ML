#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Important libraries for the code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


print ('Hello World')


# In[3]:


from sklearn import datasets
wine = datasets.load_wine()
wine.keys()


# In[4]:


import this


# In[5]:


for i in wine.keys():
    print(wine[i],'/n')


# In[6]:


wine.feature_names


# In[7]:


df_wine = pd.DataFrame(wine.data,columns = wine.feature_names)
df_wine['Target'] = wine.target
df_wine.head()


# In[8]:


# x is features
# y is labels
x= df_wine.loc[:, df_wine.columns!='Target']
y = df_wine.Target
x.shape, y.shape


# In[9]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.25,shuffle = True)
xtrain.shape,xtest.shape,ytrain.shape,ytrain.shape


# In[10]:


from sklearn.linear_model import LogisticRegression
LG = LogisticRegression()
LG.fit(xtrain,ytrain)
LG.score(xtest,ytest)


# In[11]:


arr = 5*(np.arange(10))
test_score = []
train_score = []
for i in arr:
    model_loG = LogisticRegression(max_iter = i)
    model_loG.fit(xtrain,ytrain)
    test_score.append(model_loG.score(xtest,ytest))
    train_score.append(model_loG.score(xtrain,ytrain)) 
    


# In[12]:


plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Accuracy Graph')
plt.plot(arr,train_score, label = 'Training Accuracy Graph')
plt.plot(arr,test_score, label = 'Testing Accuracy Graph')
plt.legend(loc = 'best')
plt.show()


# In[13]:


#alc stands for alcohol
alc_train = xtrain[['alcohol']]
alc_test = xtest[['alcohol']]
model_alc = LogisticRegression()
model_alc.fit(alc_train,ytrain)


# In[14]:


plt.xlabel('Features')
plt.ylabel('Class')
plt.title('Logistic Model')
plt.scatter(alc_test,ytest, label = 'Origional Data Distribution')
plt.scatter(alc_test,model_alc.predict(alc_test),color= 'red', label = 'Logistic Function')
plt.legend(loc = 'best')
plt.show()


# # Naive Bayes:

# In[15]:


from sklearn.naive_bayes import GaussianNB
NBC = GaussianNB()
NBC.fit(xtrain,ytrain)
NBC.score(xtest,ytest)


# # Label Encoding
# Converting music categories into numbers:

# In[16]:


from sklearn.preprocessing import LabelEncoder
enc =  LabelEncoder()

m_cat = ['Folk','Rock','Jazz','Soft Rock','Classical']
encd = enc.fit_transform(m_cat)
encd


# In[17]:


iris = pd.read_csv('../input/iris/Iris.csv')
iris.head()


# In[18]:


xi,yi  = iris.loc[:,iris.columns != 'Species'],iris.Species
yi.head()


# In[19]:


y_enc = enc.fit_transform(yi)
y_enc


# In[20]:


from sklearn.preprocessing import MinMaxScaler

data = np.array([[1.,-1,2.],[2.,0.,0.],[0.,1.,-1]])

MMS = MinMaxScaler()
mms = MMS.fit_transform(data)

mms


# In[21]:


Xtrain = pd.read_csv('../input/genius/Genius/X_test_ed.csv')
Xtest = pd.read_csv('../input/genius/Genius/X_train_ed.csv')

Ytest = pd.read_csv('../input/genius/Genius/Y_test.csv')
Ytrain = pd.read_csv('../input/genius/Genius/Y_train.csv')

Xtrain.shape,Ytrain.shape,Xtest.shape,Ytest.shape


# In[22]:


Xtrain.head()


# In[23]:


Ytrain.head()


# In[24]:


Xtrain = Xtrain.drop(['Loan_ID'],axis = 1)
Xtest = Xtest.drop(['Loan_ID'],axis = 1)
Xtrain.head()


# In[25]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(Xtrain,Ytrain)
model.score(Xtest,Ytest)


# In[26]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[27]:



Xtrain[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']] = Xtrain[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']].apply(encoder.fit_transform)

Xtest[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']] = Xtest[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']].apply(encoder.fit_transform)


# In[28]:


ytrain = encoder.fit_transform(ytrain)
ytest = encoder.fit_transform(ytest)
ytrain


# In[29]:


model.fit(Xtrain,Ytrain)
model.score(Xtest,Ytest)


# Normalizing data

# In[30]:


Xtrain['ApplicantIncome'] = MMS.fit_transform(Xtrain['ApplicantIncome'])
Xtrain['CoapplicantIncome'] = MMS.fit_transform(Xtrain['CoapplicantIncome'])
Xtrain['LoanAmmount'] = MMS.fit_transform(Xtrain['LoanAmmount'])
Xtrain['Loan_Ammount_Term'] = MMS.fit_transform(Xtrain['Loan_Ammount_Term'])


Xtest['ApplicantIncome'] = MMS.fit_transform(Xtest['ApplicantIncome'])
Xtest['CoapplicantIncome'] = MMS.fit_transform(Xtest['CoapplicantIncome'])
Xtest['LoanAmmount'] = MMS.fit_transform(Xtest['LoanAmmount'])
Xtest['Loan_Ammount_Term'] = MMS.fit_transform(Xtest['Loan_Ammount_Term'])




# In[31]:


Xtrain.head()


# In[32]:


model.fit(Xtrain,Ytrain)
model.score(Xtest,Ytest)


# In[ ]:




