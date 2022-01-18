#!/usr/bin/env python
# coding: utf-8

# # NUMPY CAPSTONE PROJECT - BLOOD DONATION
# ![blood_donation.png](blood_donation.png)
# <p>Blood transfusion saves lives - from replacing lost blood during major surgery or a serious injury to treating various illnesses and blood disorders. Ensuring that there's enough blood in supply whenever needed is a serious challenge for the health professionals. According to <a href="https://www.webmd.com/a-to-z-guides/blood-transfusion-what-to-know#1">WebMD</a>, "about 5 million Americans need a blood transfusion every year".</p>
# <p>Our dataset is from a mobile blood donation vehicle in Taiwan.</p>
# <p>The data is stored in <code>datasets/transfusion.data</code> and it is structured according to RFMTC marketing model (a variation of RFM). 
# <p>In this project, you are going to inspect the data using Numpy.</p>

# #### IMPORTING LIBRARIES AND DATA
# 
# * Import `numpy` as np and genfromtxt as follows: `from numpy import genfromtxt`
# 
# * Call the data by using gentxt as follows: `gentxt("YourDirectory", delimiter = ","`

# In[14]:


import numpy as np

from numpy import genfromtxt
my_data = genfromtxt("C:\\Users\\HP\\Desktop\\DATA SCIENCE\\DS-47-MUNEVVER\\B47-DS-TR\\06-Python Libraries\\Numpy Capstone Projects\\datasets\\transfusion.data", delimiter = ",")

my_data


# * Inspect our data's type by `my_data`

# In[15]:


type(my_data)


# * Use `ndim` to see how many dimensions data has.

# In[16]:


my_data.ndim


# * Return the first row our data.

# In[17]:


my_data[0]


# * First row contains `nan` values. Delete `nan` values by `np.delete()`
# * Note: `nan` values are located in `0,0`

# In[18]:


my_data = np.delete(my_data,0,0)


# * Return `my_data` to check whether you removed `nan` values or not.

# In[19]:


my_data


# * To see the dimensions of the data, use `shape`

# In[20]:


my_data.shape


# * To see how many unit(eleman) you have on your data, use `size`

# In[21]:


my_data.size


# * To see the data type inside `my_data`, use `dtype`

# In[22]:


my_data.dtype


# * To see the size of the each unit(eleman), use `itemsize`

# In[23]:


my_data.itemsize


# * Create a matrix that has 2 rows and 5 columns and contains 0 by `np.zeros`. Name it as `sifir`

# In[24]:


sifir = np.zeros((2,5))
sifir


# * Create a matrix that has 2 rows and 5 columns and contains 1 by `np.ones`. Name it as `bir`

# In[25]:


bir = np.ones((2,5))
bir


# * Create a matrix that has 2 rows and 5 columns and contains 38 by `np.full`. Name it as `otuzsekiz`

# In[26]:


otuzsekiz = np.full((2,5),38)
otuzsekiz


# * Create an eye matrix that has 5 rows and 5 columns by `np.eye`. Name it as `eye`

# In[27]:


eye = np.eye(5)
eye


# * Create a matrix that has 2 rows and 5 columns and contains random values between 0 and 1 by `np.random.random`. Name it as `random`

# In[28]:


random = np.random.random((2,5))


# * Create a matrix that has 2 rows and 5 columns(use `reshape` for that) and contains values increases 1 at a time, and between 1 and 10 by `np.linspace`. Name it as `linsp`

# In[29]:


linsp = np.linspace(1,10,10).reshape((2,5)) 
linsp


# * Extract `linsp` with `np.sqrt` and name the result as `linsp`

# In[30]:


linsp = np.sqrt(linsp)
linsp


# * exponentiate `random` and name the result as `random`

# In[31]:


random = random ** 2
random


# * Sum `linsp` and `random` and name it as `toplam`

# In[32]:


toplam = linsp + random 
toplam


# * Divide `bir` and `sifir` and name it as `bolme`
# * If you receive and warning or error, briefly explain why

# In[33]:


bolme1 = bir /sifir

bolme1


# * Subtract `bir` and `sifir` and name it as `cikarma`

# In[35]:


cikarma = bir - sifir
cikarma


# * divide  `cikarma` and `toplam`. Then, name it as `bolme`

# In[36]:


bolme = cikarma / toplam
bolme


# * Multiply `toplam` and `bolme` by element basis and name it as `ecarpma`

# In[38]:


ecarpma = toplam * bolme
ecarpma


# * Multiply `ecarpma` and `eye` by matrix basis and name it as `mcarpma`

# In[40]:


mcarpma = ecarpma @ eye
mcarpma


# * Create matrix `a` that has following values:
# 
# `[[ 1 2 3 4 5]
#   [ 6 7 8 9 10]]`

# In[41]:


a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
a


# * Return the **boolean values** result of the values that are more than 3

# In[42]:


a>3


# * Return the values that are more than 3

# In[44]:


a[a>3]


# * Set the values that are more than 3 to 0 and name the result as `a`

# In[45]:


a[a>3] = 0
a


# * Join `a` and `mcarpma ` by using stack functions(`axis=1`) and name it as `stc`

# In[46]:


stc = np.stack((a,mcarpma),axis=1)
stc


# * Take the 1'st and 3'rd rows from `stc`, assign them to a new matrix.Name this new matrix as `guncel`

# In[47]:


guncel = stc[:,0:1]
guncel


# * Make guncel 2 dimensional array.

# In[48]:


guncel= guncel.reshape(2,5)
guncel.ndim


# * Do you remember the `my_data` that we defined above?
# * Join `my_data` with `guncel` by using `concatenate` method vertically(alt alta). Name the result as `data`

# In[49]:


data = np.concatenate([my_data,guncel])
data


# * Sum the columns of `data`

# In[50]:


data.sum(axis=0)


# * Sum the rows of `data`

# In[51]:


data.sum(axis=1)


# * Return the maximum values of each column

# In[52]:


data.max(axis=0)


# * Return the maximum values of each row

# In[53]:


data.max(axis=1)


# * Return the minimum values of each column

# In[54]:


data.min(axis=0)


# * Return the minimum values of each row

# In[55]:


data.min(axis=1)


# * Find the index of the biggest value
# * Note: The value you're about to reach is the index of our `data`'s flatten value.

# In[56]:


np.argmax(data)


# * Find the index of the smallest value

# In[57]:


np.argmin(data)


# * Transpose to `data` and set it the result as `datat`

# In[58]:


datat = data.T
datat


# In[ ]:




