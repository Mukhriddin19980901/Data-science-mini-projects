#!/usr/bin/env python
# coding: utf-8

# In[8]:


# standart deviation formulasi 
from math import sqrt
import numpy as np
age =np.array ([81,82,88,86,91,85,86])
mse=0
dev = 0
# mean(x^) = sum(a)/len(a)
#variance - (i-sum(a)/len(a))**2/len(a)-1
# standart deviation σ =sqrt( variance)
x_=sum(age)/len(age)
for i in age:
    mse+=((i-(x_))**2)
dev=(-1*mse/len(age))
if dev<0:
    dev = abs(dev)
print(sqrt(dev))


# In[9]:


np.std(age)

