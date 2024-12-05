#!/usr/bin/env python
# coding: utf-8

# # AI Lab-2
# ## Introduction to Pandas
# ### Author : Abdul Basit

# #### Lab Tasks

# ### 1. Load the Titanic Dataset in Kaggle notebook from given link below:
# 
# https://www.kaggle.com/hesh97/titanicdataset-traincsv
# 
# Use the bar plot to show following plots:
# 
# > Plot the Number of people survived and did not survive. Hint: Plot the counts values of “Survived” column of dataset.
# 
# > Also plot survived comparison by class and by gender
# 
# > Use the “groupby” function of Pandas to group the mean values (of all features) of passengers’’ gender and class.
# ______________________________________________________________________________
# ### 3. Explore how to add and delete column from a data frame, explain and illustrate with an example. Also define concept of axis in pandas.
# 
# > Generate a random data frame of 20 rows and 04 columns and change their columns names.
# 
# > Apply logical conditions to print values greater than 0.5 in any one column.
# 
# > Explore sort function in Pandas, apply ascending and descending sorting according.
# 
# > Loc and iloc are two functions of pandas for accessing the data from the data frame. Define the difference between them and use them each in at least 2 examples.

# In[17]:


#importing some impotant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# Titanic dataset link
# https://www.kaggle.com/abbasitc/titanic-dataset

df = pd.read_csv('../input/titanic-dataset/Titanic_dataset.csv')
df .head()


# # AI lab-3 exercise-1

# Question 1.1

# In[2]:


#Plot for those who survived and those who didn't survived
sb.countplot(data=df,x='Survived')


# # Question 1.2

# In[3]:


# Plot by gender wise for survivors
sb.countplot(x='Survived',data=df,palette='Blues',hue='Sex')


# # Question 1.3

# In[4]:


# Plot grouped by gender and class wise

sb.countplot(x='Survived',data=df,palette='rainbow',hue='Pclass')

df.groupby('Sex').mean()

df.groupby('Pclass').mean()


# # AI lab-3 exercise-2

# # Question 2.1

# # Axis in Pandas
# Data frames can be thought of as having two dimensions, or "axes" the rows of the DataFrame, and the columns. In pandas, these are referred to as axis 0 and axis 1. Axis 0 refers to the rows, and axis 1 refers to the columns. 

# In[5]:


import pandas as pd
a1=pd.DataFrame({'A':[4,6,4,2],'B':[9,5,7,8],'C':[5,8,3,9]})
a2=pd.DataFrame({'A':[9,3,8,4],'B':[5,9,6,9],'C':[9,6,7,8]})
a3=pd.DataFrame()

a1,a2,a3


# In[7]:


#Adding column in a3
a3=a1.append(a1,ignore_index = True)
a3


# In[8]:


# Deleting a colunm from the dataframe
a2.pop ('B')
a2


# # Question 2.2

# In[9]:


# Generating random numbers by pandas
import numpy as np
r_num = pd.DataFrame(np.random.rand(20,4))
r_num


# In[10]:


# Changing names of columns
r_num.columns=['P','Q','R','S']
r_num


# # Question 2.3

# In[11]:


# Printing numbers greater than 0.5
r_num[r_num>0.5]


# # Question 2.4

# In[12]:


# Sorting the data in ascending using pandas function it sort by default in ascending order
r_num.sort_values(by=['P', 'Q','R','S'])


# In[13]:


# Sorting the data in descending order using pandas function
r_num.sort_values(by=['P','Q','R','S'], ascending=False)


# # Loc and iloc
# Loc and iloc both functions are used in pandas for slicing. Loc function is used to access any specified row using row's name as string argument for loc function, Where as iloc takes integer argument as input(i.e index number of the row). 

# In[14]:


# Creating data for this example
data_frame = pd.DataFrame({'A': [11, 21, 31],
                   'B': [12, 22, 32],
                   'C': [13, 23, 33]},
                  index=['ONE', 'TWO', 'THREE'])
data_frame


# In[15]:


# Accessing the second row using loc function
data_frame.loc['TWO']


# In[16]:


# Accessing the second row using iloc function
data_frame.iloc[1]


# In[ ]:




