#!/usr/bin/env python
# coding: utf-8

# # AI Lab-1
# ## Introduction to Python
# ### Author : Abdul Basit

# ### Lab Practice

# In[46]:


print ('Hello ABC')
abc = 45.5
print(type(abc))


# In[47]:


b = 'Hi Abdul'
type(b)


# In[48]:


my_friends = ['Ali', 'Wali', 'Asad', 'Jameel','Fayaz']
print (my_friends[0])
my_friends[0:3]


# In[49]:


print(my_friends[4:])


# In[50]:


print(my_friends.index('Fayaz'), my_friends[4])


# In[51]:


mydata = ['Abdul', 23, 'Good']


# In[52]:


print(type(mydata[-2]))


# In[53]:


l1 = [1,3,5,7,9]
l2 = [2,4,6,8,10]


# In[54]:


newl = [l1+l2]


# In[55]:


print(newl)


# In[56]:


newl2 = l1 + l2


# In[57]:


print (newl2)


# In[58]:


l = []
l.append(l1)
print(l)


# In[59]:


l.append(l2)
print(l)


# In[60]:


print(my_friends)


# In[61]:


my_friends.append('Abdul Basit')


# In[62]:


print(l)


# In[63]:


l1.extend(l2)
print(l1)


# In[64]:


my_friends.sort()
print(my_friends)


# In[65]:


print(my_friends.count('Ali'))


# In[66]:


print(l)


# In[67]:


l.clear()
print(l)


# In[68]:


print(l1.copy())


# In[69]:


l1.insert(0,55)
print(l1)


# In[70]:


l1.remove(55)
print(l1)


# In[71]:


l1.insert(1, 'Orange')
print(l1)


# In[72]:


l1.pop(1)
print(l1)


# In[73]:


l1.remove(10)
print(l1)


# In[74]:


l1.reverse()
print(l1)


# In[75]:


l1.sort()
print(l1)


# In[88]:


print(type(l1))


# In[77]:


print(my_friends)


# In[78]:


type(my_friends)


# In[79]:


my = ('A', 'B', 'C')
type(my)


# In[80]:


print(sum(l1))


# In[81]:


print(max(l1))


# In[82]:


min(l1)
print(l1)


# In[83]:


print(list(l1))


# In[84]:


# For Loop
count = 0
nf = 0

for i in range(10):
    print(i, ": is a number")
    count+=1
nf =  count + 1
print('Number in For loop = ',count,"Number not in for loop", nf)


# In[85]:


# While Loop

k = 0 
while True:
    k += 3
    if k>36:
        print('Limit number has come')
        break
      
    else:
        print(k, "'Waiting for Limit Number'")
print(k)    


# In[86]:


# Definning a function for BMI calculator
def BMI(height,weight):
    height_m = height/3.281
    print('Your BMI is = ', weight/height_m**2)
BMI(6,65)    


# In[87]:


# Definning a function for square root

def sq(num):
     n = num**2
     print('Square is = ', n)
sq(3)        

