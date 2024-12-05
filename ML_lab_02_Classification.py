#!/usr/bin/env python
# coding: utf-8

# ### ML Lab-02
# # Exploring logistic regression, LDA, and kNN
# ## Abdul Basit

# ### Tasks
# ##### Load iris dataset from scikit-learn.
# 1. Divide the dataset into training and test sets
# 2. Train LR, LDA, and kNN-3
# 3. Determine classification accuracies of all the classifiers
# 4. Discuss the classifier that performed the best and the worst
# 5. Repeat steps 2 – 4 for the data with (a) two classes and all features, (b) two features and all classes, and (c) equal train-test ratio and all features and classes

# In[18]:


#important Libraraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris


# In[19]:


iris = load_iris()


# In[20]:


iris.target


# In[21]:


iris.keys()


# In[29]:


iris.target_names


# In[22]:


iris.feature_names


# In[31]:


x = iris.data
y = iris.target


# In[32]:


iris.target


# In[33]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()


# In[34]:


df.plot()


# In[35]:


#df.plot.scatter(x='sepal length (cm)', y = 'sepal width (cm)')


# In[36]:


#Spliting data into train test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[37]:


# Training Model for logestic regression
# We can face max_iterationn problem so try to increase max iterationns
# LogisticRegression(max_iter = 500)

logreg = LogisticRegression(max_iter = 500)
logreg.fit(x_train, y_train)


# In[38]:


yprelr = logreg.predict(x_test)


# In[39]:


print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))


# In[40]:


#taining model for LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)


# In[41]:


yprelda = lda.predict(x_test)


# In[42]:


print('Accuracy of logistic LDA classifier on test set: {:.2f}'.format(lda.score(x_test, y_test)))


# In[43]:


sepal = df['sepal length (cm)']


# In[44]:


plt.hist(sepal, color = 'blue', edgecolor = 'blue', bins = 40)
plt.show()


# In[45]:


#Training KNN
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train, y_train)


# In[46]:


ypreknn = KNN.predict(x_test)


# In[51]:


from sklearn.metrics import accuracy_score
lg_acc = accuracy_score(yprelr, y_test)
lda_acc = accuracy_score(yprelda, y_test)
knn_acc = accuracy_score(ypreknn, y_test)


# In[48]:


#finding the co-efficients/Betas in the data
print(logreg.coef_)
print(lda.coef_)
# Note KNN has no coefficietns
# This command will not work "KNN.coef_"


# In[49]:


print(yprelr)
print(yprelda)
print(ypreknn)


# In[52]:


#Displying Classes Accuricies
print(lg_acc)
print(lda_acc)
print(knn_acc)


# In[ ]:




