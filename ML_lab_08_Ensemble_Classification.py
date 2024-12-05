#!/usr/bin/env python
# coding: utf-8

# ### ML Lab-08
# # Ensemble Classification Methods
# ## Abdul Basit

# ### Tasks
# ##### Load iris dataset from scikit-learn.
# Download the “Iris” dataset from scikit-learn dataset module and select only two features, “sepal length” and “petal width” and two classes, "setosa" and “versicolor”. Clean the data if required. Divide the data samples into train and test sets.
# 1. Train and test a decision tree classifier and report the achieved accuracies.
# 2. Repeat step 1 for bagging classifier using: model = BaggingClassifier(base_estimators = tree, n_estimators = 100, max_samples = 1.0, bootstrap = True, bootstrap_features = False, ran-dom_state = 1).
# 3. Apply boosting (using AdaBoost) on the same dataset using: model1 = AdaBoostClassi-fier(base_estimators = tree, n_estimators = 100, learning_rate = 0.1, random_state = 1).
# 4. Compare and comment the accuracies achieved in step 1 – 3.

# In[2]:


#important Libraraies
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


# In[3]:


iris = load_iris()


# In[5]:


iris.keys()


# In[6]:


iris.target_names


# In[187]:


data = iris.data
col = iris.target


# In[188]:


data = pd.DataFrame(iris.data, columns=iris.feature_names)
data.head()


# In[189]:


X= data[['sepal length (cm)','petal width (cm)']]
X.head()


# In[190]:


X['labels'] =  col
X


# In[191]:


X.shape


# In[195]:


new_data = X[X.labels != 2]
new_data.head()


# In[196]:


new_data.shape


# In[197]:


X = new_data[['sepal length (cm)','petal width (cm)']]
X[:5]


# In[198]:


y = new_data[['labels']]
y.shape


# In[199]:


#Spliting data into train test set
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=24)


# In[200]:


tree = DecisionTreeClassifier(criterion = 'entropy',random_state = 1, max_depth = None)
bag = BaggingClassifier(base_estimator = tree,
                      n_estimators = 10,
                       max_samples = 1.0,
                       max_features = 1.0,
                       bootstrap = True)


# In[201]:


DTR = DecisionTreeRegressor(criterion='mse')
DTR.fit(x_train, y_train)


# In[202]:


y_predDTR = DTR.predict(x_test)
print('Accuracy of Decision Tree Regressor on test set: {:.2f}'.format(DTR.score(x_test, y_test)))


# In[203]:


DTC = DecisionTreeClassifier(criterion = 'entropy',random_state = 1, max_depth = None)
DTC.fit(x_train, y_train)


# In[204]:


y_predDTC = DTC.predict(x_test)
print(y_predDTC)


# In[205]:


print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(DTC.score(x_test, y_test)))


# In[206]:


# Finding accuracy score
# accuracy_score(labels_test,pred)
DTC_acc_scr = accuracy_score(y_test,y_predDTC)
DTC_acc_scr


# In[207]:


Bagging = BaggingClassifier(base_estimator=tree,n_estimators = 100, max_samples = 1.0, bootstrap = True, bootstrap_features = False, random_state = 1)
Bagging.fit(X, y)


# In[208]:


Bagging_pred = Bagging.predict(x_test)
Bagging_pred 


# In[209]:


Bagging_acc_scr = accuracy_score(y_test,Bagging_pred)
Bagging_acc_scr


# In[210]:


ABC = AdaBoostClassifier(base_estimator=tree,n_estimators = 100, learning_rate = 0.1, random_state = 1)
ABC.fit(X, y)


# In[211]:


ABC_pred = ABC.predict(x_test)
ABC_pred 


# In[212]:


ABC_acc_scr = accuracy_score(y_test, ABC_pred)
ABC_acc_scr


# In[213]:


tree = DecisionTreeClassifier(criterion = 'entropy',random_state = 1, max_depth = None)
bag = BaggingClassifier(base_estimator = tree,
                      n_estimators = 10,
                       max_samples = 1.0,
                       max_features = 1.0,
                       bootstrap = True)


# In[214]:


cv = cross_val_score(tree,iris.data, iris.target, cv = 10)
print(cv)


# In[215]:


tree_model = tree.fit(x_train,y_train)
Y_tree_pred = tree_model.predict(x_test)
print(Y_tree_pred)


# In[216]:


tree_acc_scr = accuracy_score(Y_tree_pred,y_test)
print(tree_acc_scr)


# In[217]:


bag_model = bag.fit(x_train,y_train)
Y_bag_pred = bag_model.predict(x_test)
print(Y_bag_pred)


# In[218]:


bag_acc_scr = accuracy_score(y_test,Y_bag_pred)
print(bag_acc_scr)


# In[219]:


fet = iris.data[:,[2,3]]
tar  = iris.target


# In[ ]:




