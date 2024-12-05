#!/usr/bin/env python
# coding: utf-8

# # AI Lab-1
# 
# ## Introduction to Python
# 
# ### Author : Abdul Basit

# # **Solution for lab-1 exercise-1**

# In[1]:


#creating list named names and height
names = ['Akhatar', 'Qadeer', 'Nabeel', 'Fayaz', 'Rameez']
print (names)
height = [5.7, 6, 6.1, 5.5, 5.6]
height


# In[2]:


#finding length of names and height lists
a = len(names)
b = len(height)

print ("Number of entries in names list:",a )
print ("Number of entries in Height list:",b)


# In[3]:


#finding maximum height of friend from names
y = max(height)
print(y)
z = height.index(y)
print (names[z])

#output will be maximum Height of Friend and his name


# In[4]:


#finding minimum height of friend from names
y = min(height)
print(y)
z = height.index(y)
print (names[z])

#output will be minimum Height of Friend and his name


# In[5]:


# Sorting names in list
names.sort()
names


# In[6]:


#finding average height of Friends
average = sum(height)/len(height)
average


# # **Solution to lab-1 Exercise-2**

# In[7]:


def BMI(height,weight):      # Creaing function named BMI
    result = (703) * (weight/(height**2))  # Calculating the BMI using Formulla
    # Checking all the condition of BMI using if-else conditions
    if result< 18.5:            
        print ('You are underweigt')
    elif result < 24.9:
        print ('You are normal')
    elif result < 29.9:
        print ('You are overweigt')
    else:
        print ('You are obesity')
        print ("Don't worry input your height in cm and try again") # Note for if enterd wrong height
        
BMI(164,55)    # Calling the functions


# # **Solution to Lab-1 exercise-3**

# In[8]:


def ctof(c):   #creating a functon to convert celcius to Farhneit
    temperature = (c * (9/5)) + 2  # Using arthimetic Formulla to convert
    print ("Today's temperature is", temperature , "f.") # Displaying the results
    
ctof(34)    # Calling the function 
    


# # **Soultion to Lab-1 exercise-4**

# In[9]:


#creating two lists

list1 = [] #list1 for even numbers
list2 = [] #list2 for odd numbers

for i in range(100): 
    
    if i%2 == 0: # Using modulo operator to check weather number is even or not
        list1.append(i) # appending even numbers in list1
    else:
        list2.append(i) # appending odd numbers in list2
    
print ('List of even numbers:')        
print (list1)
print ('List of odd numbers:') 
print (list2)

print ('Adding both lists together') 
newlist = list1 + list2
print (newlist)

print ("Sorting the new created list:")
newlist.sort()
print(newlist)
    
    

