#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mimg


# In[33]:


image = cv2.imread("amazon.jpg")
image = cv2.resize(image,dsize=(300,400),interpolation=cv2.INTER_AREA)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image)


# In[37]:


gray_blur = cv2.GaussianBlur(image,(5,5),0)
#print(gray_blur)
plt.figure(figsize=(10,20))

plt.subplot(121)

plt.title("gaussain")
plt.imshow(gray_blur,cmap='gray')
plt.subplot(122)
plt.title("original")
plt.imshow(image,cmap='gray')


# In[ ]:


def four(rasm):
    f = np.fft.fft2(rasm)
    fshift = np.fft.fftshift(f)
    chastota = 20 * np.log(np.abs(fshift))
    return chastota


# In[4]:


rasm1gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
norm_rasm1gray = rasm1gray / 255.0
image = four(norm_rasm1gray)
plt.imshow(image,cmap='gray')


# In[32]:


# 
kernel = np.array([[-1,-1,1,1],
                   [-1,-1,1,1],
                   [-1,-1,1,1],
                   [-1,-1,1,1]])  # Vertical

kerne1 = np.array([[-1,-1,-1,-1],
                   [-1,-1,-1,-1],
                   [1,1,1,1],
                   [1,1,1,1]])  # Horizontal

sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

sobel_y = np.array([[-1,-1,-1],
                    [0,0,0],
                    [1,1,1]])
rasm = cv2.imread("amazon.jpg")
rasm_gray = cv2.cvtColor(rasm,cv2.COLOR_BGR2GRAY)
print(rasm_gray)
plt.subplot(211)
plt.imshow(rasm_gray,cmap='gray')
filter_rasm = cv2.filter2D(rasm_gray,-1,sobel_x)
plt.subplot(212)
plt.imshow(filter_rasm,cmap = 'rainbow')


# In[6]:


filter_rasm = cv2.filter2D(rasm_gray,-1,sobel_y)
plt.imshow(filter_rasm,cmap='gray')


# In[16]:


get_ipython().run_line_magic('matplotlib', 'qt')
_, rasm_ikki = cv2.threshold(filter_rasm,21,255,cv2.THRESH_BINARY)
plt.imshow(rasm_ikki,cmap='gray')


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
image = cv2.imread("C:/Users/USER/20-dars OpenCV/1-parij.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image1 = cv2.imread("C:/Users/USER/20-dars OpenCV/haland.jpg",0)
image1 [image1==0] = 255
#image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(image,cmap ="gray")
plt.subplot(122)
plt.imshow(image1,cmap ="gray")


# In[40]:


rasm = cv2.imread("A/A0001_test.jpg")
rasm_gray = cv2.cvtColor(rasm,cv2.COLOR_BGR2GRAY)
plt.subplot(121)
print(rasm_gray)
plt.imshow(rasm_gray,cmap='gray')
plt.subplot(122)
plt.imshow(rasm1,cmap = 'g


# In[50]:


image = cv2.imread("A/A0001_test.jpg")
rasm1gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
plt.subplot(461)
plt.figure(figsize=(50,80))
plt.imshow(rasm1gray)
norm_rasm1gray = rasm1gray / 255.0
image = four(norm_rasm1gray)
plt.subplot(462)
plt.imshow(image,cmap='gray')

kernel_rasm = cv2.filter2D(rasm1gray,-1,kernel)
plt.subplot(463)
plt.imshow(kernel_rasm,cmap='gray')

kerne1_rasm = cv2.filter2D(rasm1gray,-1,kerne1)
plt.subplot(464)
plt.imshow(kerne1_rasm,cmap='gray')

sobelx_rasm = cv2.filter2D(rasm1gray,-1,sobel_x)
plt.subplot(465)
plt.imshow(sobelx_rasm,cmap='gray')

sobely_rasm = cv2.filter2D(rasm1gray,-1,sobel_y)
plt.subplot(466)
plt.imshow(sobely_rasm,cmap='gray')


# In[5]:


gs = (1/9)*np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])

laplacian = np.array([[0,1,0],
                   [1,-4,1],
                   [0,1,0]])  # Horizontal

sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

sobel_y = np.array([[-1,-1,-1],
                    [0,0,0],
                    [1,1,1]])


# In[1]:


image = cv2.imread("TINIQQ.png")
rasm = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
plt.figure(figsize=(50,50))
plt.subplot(521)
plt.title('org')
plt.imshow(rasm)
rasm1gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

norm_rasm1gray = rasm1gray / 255.0
image = four(norm_rasm1gray)
plt.subplot(522)
plt.title("fft")
plt.imshow(image,cmap='gray')

gs_rasm = cv2.filter2D(rasm1gray,-1,gs)
plt.subplot(523)
plt.title("gs")
plt.imshow(gs_rasm,cmap='gray')

image_gs = four(gs_rasm)
plt.subplot(524)
plt.title("gs_fft")
plt.imshow(image_gs,cmap='gray')


lap_rasm = cv2.filter2D(rasm1gray,-1,laplacian)
plt.subplot(525)
plt.title("lap")
plt.imshow(lap_rasm,cmap='gray')

image_lap = four(lap_rasm)
plt.subplot(526)
plt.title("lap_fft")
plt.imshow(image_lap,cmap='gray')


sobelx_rasm = cv2.filter2D(rasm1gray,-1,sobel_x)
plt.subplot(527)
plt.title("sobx")
plt.imshow(sobelx_rasm,cmap='gray')

image_sobx = four(sobelx_rasm)
plt.subplot(528)
plt.title("sobx fft")
plt.imshow(image_sobx,cmap='gray')


sobely_rasm = cv2.filter2D(rasm1gray,-1,sobel_y)
plt.subplot(529)
plt.title("soby")
plt.imshow(sobely_rasm,cmap='gray')

image_soby = four(sobely_rasm)
plt.subplot(5,2,10)
plt.title("soby_fft")
plt.imshow(image_soby,cmap='gray')


# In[76]:


D = cv2.imread("Dimage.png")
plt.subplot(121)
plt.imshow(D)
D = cv2.cvtColor(D,cv2.COLOR_RGB2BGR)
Dim  = four(D)
plt.subplot(122)
plt.title("fft")
plt.imshow(Dim,cmap='gray')


# In[82]:


# 
D = cv2.imread("G/G0001_test.jpg")
plt.subplot(121)
plt.imshow(D)
blur =(1/9)*np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]]) 
rasm = cv2.filter2D(D,-1,blur)
plt.subplot(122)
plt.imshow(rasm)


# In[ ]:




