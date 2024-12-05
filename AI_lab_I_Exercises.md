# AI Lab-1

## Introduction to Python

### Author : Abdul Basit

# Lab Exercise tasks

### 1. Define “names” and “height” lists that contain name and height of your friends respectively. Use the python packages to convert the names list in ascending order. Find the total number of entries in list, maximum height, and minimum height with their names and find the average height of your friends.


###  2. Develop a BMI function in python programming language that gets the weight and height as inputs and show that whether input user is underweight, normal weight, overweight or obese.


###  3. Write a function that takes input a number, which is temperature in degree Celsius and return the temperature in kelvin and Fahrenheit.


###  4. Create a list of even numbers and odd numbers with the help of for loop and append function. Than combine both lists together.

# **Solution for lab-1 exercise-1**


```python
#creating list named names and height
names = ['Akhatar', 'Qadeer', 'Nabeel', 'Fayaz', 'Rameez']
print (names)
height = [5.7, 6, 6.1, 5.5, 5.6]
height
```

    ['Akhatar', 'Qadeer', 'Nabeel', 'Fayaz', 'Rameez']
    




    [5.7, 6, 6.1, 5.5, 5.6]




```python
#finding length of names and height lists
a = len(names)
b = len(height)

print ("Number of entries in names list:",a )
print ("Number of entries in Height list:",b)
```

    Number of entries in names list: 5
    Number of entries in Height list: 5
    


```python
#finding maximum height of friend from names
y = max(height)
print(y)
z = height.index(y)
print (names[z])

#output will be maximum Height of Friend and his name

```

    6.1
    Nabeel
    


```python
#finding minimum height of friend from names
y = min(height)
print(y)
z = height.index(y)
print (names[z])

#output will be minimum Height of Friend and his name
```

    5.5
    Fayaz
    


```python
# Sorting names in list
names.sort()
names
```




    ['Akhatar', 'Fayaz', 'Nabeel', 'Qadeer', 'Rameez']




```python
#finding average height of Friends
average = sum(height)/len(height)
average
```




    5.779999999999999



# **Solution to lab-1 Exercise-2**


```python
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
```

    You are underweigt
    

# **Solution to Lab-1 exercise-3**


```python
def ctof(c):   #creating a functon to convert celcius to Farhneit
    temperature = (c * (9/5)) + 2  # Using arthimetic Formulla to convert
    print ("Today's temperature is", temperature , "f.") # Displaying the results
    
ctof(34)    # Calling the function 
    
```

    Today's temperature is 63.2 f.
    

# **Soultion to Lab-1 exercise-4**


```python
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
    
    
```

    List of even numbers:
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]
    List of odd numbers:
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
    Adding both lists together
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
    Sorting the new created list:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    
