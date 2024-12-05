# AI Lab-1
## Introduction to Python
### Author : Abdul Basit

### Lab Practice


```python
print ('Hello ABC')
abc = 45.5
print(type(abc))
```

    Hello ABC
    <class 'float'>
    


```python
b = 'Hi Abdul'
type(b)
```




    str




```python
my_friends = ['Ali', 'Wali', 'Asad', 'Jameel','Fayaz']
print (my_friends[0])
my_friends[0:3]
```

    Ali
    




    ['Ali', 'Wali', 'Asad']




```python
print(my_friends[4:])
```

    ['Fayaz']
    


```python
print(my_friends.index('Fayaz'), my_friends[4])
```

    4 Fayaz
    


```python
mydata = ['Abdul', 23, 'Good']
```


```python
print(type(mydata[-2]))
```

    <class 'int'>
    


```python
l1 = [1,3,5,7,9]
l2 = [2,4,6,8,10]
```


```python
newl = [l1+l2]
```


```python
print(newl)
```

    [[1, 3, 5, 7, 9, 2, 4, 6, 8, 10]]
    


```python
newl2 = l1 + l2
```


```python
print (newl2)
```

    [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    


```python
l = []
l.append(l1)
print(l)
```

    [[1, 3, 5, 7, 9]]
    


```python
l.append(l2)
print(l)
```

    [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]
    


```python
print(my_friends)
```

    ['Ali', 'Wali', 'Asad', 'Jameel', 'Fayaz']
    


```python
my_friends.append('Abdul Basit')
```


```python
print(l)
```

    [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]
    


```python
l1.extend(l2)
print(l1)
```

    [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    


```python
my_friends.sort()
print(my_friends)
```

    ['Abdul Basit', 'Ali', 'Asad', 'Fayaz', 'Jameel', 'Wali']
    


```python
print(my_friends.count('Ali'))
```

    1
    


```python
print(l)
```

    [[1, 3, 5, 7, 9, 2, 4, 6, 8, 10], [2, 4, 6, 8, 10]]
    


```python
l.clear()
print(l)
```

    []
    


```python
print(l1.copy())
```

    [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    


```python
l1.insert(0,55)
print(l1)
```

    [55, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    


```python
l1.remove(55)
print(l1)
```

    [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    


```python
l1.insert(1, 'Orange')
print(l1)
```

    [1, 'Orange', 3, 5, 7, 9, 2, 4, 6, 8, 10]
    


```python
l1.pop(1)
print(l1)
```

    [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
    


```python
l1.remove(10)
print(l1)
```

    [1, 3, 5, 7, 9, 2, 4, 6, 8]
    


```python
l1.reverse()
print(l1)
```

    [8, 6, 4, 2, 9, 7, 5, 3, 1]
    


```python
l1.sort()
print(l1)
```

    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    


```python
print(type(l1))
```

    <class 'list'>
    


```python
print(my_friends)
```

    ['Abdul Basit', 'Ali', 'Asad', 'Fayaz', 'Jameel', 'Wali']
    


```python
type(my_friends)
```




    list




```python
my = ('A', 'B', 'C')
type(my)
```




    tuple




```python
print(sum(l1))
```

    45
    


```python
print(max(l1))
```

    9
    


```python
min(l1)
print(l1)
```

    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    


```python
print(list(l1))
```

    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    


```python
# For Loop
count = 0
nf = 0

for i in range(10):
    print(i, ": is a number")
    count+=1
nf =  count + 1
print('Number in For loop = ',count,"Number not in for loop", nf)
```

    0 : is a number
    1 : is a number
    2 : is a number
    3 : is a number
    4 : is a number
    5 : is a number
    6 : is a number
    7 : is a number
    8 : is a number
    9 : is a number
    Number in For loop =  10 Number not in for loop 11
    


```python
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
```

    3 'Waiting for Limit Number'
    6 'Waiting for Limit Number'
    9 'Waiting for Limit Number'
    12 'Waiting for Limit Number'
    15 'Waiting for Limit Number'
    18 'Waiting for Limit Number'
    21 'Waiting for Limit Number'
    24 'Waiting for Limit Number'
    27 'Waiting for Limit Number'
    30 'Waiting for Limit Number'
    33 'Waiting for Limit Number'
    36 'Waiting for Limit Number'
    Limit number has come
    39
    


```python
# Definning a function for BMI calculator
def BMI(height,weight):
    height_m = height/3.281
    print('Your BMI is = ', weight/height_m**2)
BMI(6,65)    
```

    Your BMI is =  19.436735138888892
    


```python
# Definning a function for square root

def sq(num):
     n = num**2
     print('Square is = ', n)
sq(3)        
```

    Square is =  9
    
