#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mimg


# In[10]:


def four(rasm):
    f = np.fft.fft2(rasm)
    fshift = np.fft.fftshift(f)
    chastota = 20 * np.log(np.abs(fshift))
    return chastota


# In[11]:


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
gs = (1/9)*np.array([[1,1,1],    #GaussainBlur
                   [1,1,1],
                   [1,1,1]])

laplacian = np.array([[0,1,0], # Laplacian Blur
                   [1,-4,1],
                   [0,1,0]])


# In[12]:


rasm = cv2.imread ("miyya.jpg")
plt.imshow(rasm)


# In[13]:


kengroq = cv2.Canny(rasm, 100, 200)
torroq  = cv2.Canny(rasm, 240, 270)
plt.figure(figsize=(20,10))
plt.subplot(121)
plt.imshow(kengroq,cmap='gray')
plt.subplot(122)
plt.imshow(torroq,cmap='gray')


# In[14]:


img = cv2.imread("20-Dars OpenCV/car_green.jpg")
lines = cv2.HoughLinesP(torroq,1,np.pi/180,80,np.array([]),20,10)
print(lines)
chiziq_rasm = np.copy(img)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(chiziq_rasm,(x1,y1),(x2,y2),(0,0,255),2)
plt.imshow(chiziq_rasm)


# In[15]:


rasm = cv2.imread("circle.jpg")
rasm = cv2.cvtColor(rasm,cv2.COLOR_BGR2RGB)
rasm2 = cv2.cvtColor(rasm,cv2.COLOR_BGR2GRAY)
rasm2 = cv2.GaussianBlur(rasm2,(3,3),0)
plt.imshow(rasm2,cmap='gray') 
rasm_copy = np.copy(rasm)

aylana = cv2.HoughCircles(rasm2,cv2.HOUGH_GRADIENT,1,
                         minDist=30,
                         param1=30,
                         param2=12,
                         minRadius=15,
                         maxRadius=30)

aylana = np.uint16(np.around(aylana))

for i in aylana[0,:]:
    cv2.circle(rasm_copy,(i[0],i[1]),i[2],(0,0,255),3)
    
plt.imshow(rasm_copy)


# In[16]:


rasm = cv2.imread('harf.png')
rasm = cv2.cvtColor(rasm,cv2.COLOR_BGR2GRAY)
plt.imshow(rasm,cmap='gray')


# In[17]:


kernel = np.ones((7,7),np.uint8)
dilate = cv2.dilate(rasm,kernel,iterations = 1 )
plt.imshow(dilate,cmap = "gray")


# In[18]:


erosion = cv2.erode(rasm,kernel,iterations = 2)
plt.imshow(erosion,cmap = "gray")


# In[19]:


mushuk = cv2.imread("mushukla.png")
mushuk = cv2.cvtColor(mushuk,cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.imshow(mushuk)
gray = cv2.cvtColor(mushuk,cv2.COLOR_RGB2GRAY)
_,binary = cv2.threshold(gray,254,255,cv2.THRESH_BINARY_INV)
erosion = cv2.erode(binary,kernel, -8)
plt.subplot(122)
plt.imshow(erosion,cmap='gray')


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
contours, boglama = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

chizish = np.copy(mushuk)
chizish = cv2.cvtColor(chizish,cv2.COLOR_BGR2RGB)
chizish = cv2.drawContours(chizish,contours,-1,(0,0,255),3)
chizish = cv2.cvtColor(chizish,cv2.COLOR_BGR2RGB)
plt.title("Mushuklar")
plt.imshow(chizish)
print(len(contours))


# In[21]:


burchak = []
for i in contours:
    (x,y),(Ma,ma),angle = cv2.fitEllipse(i)
    burchak.append(angle)
print(burchak)


# In[22]:


x,y,w,h = cv2.boundingRect(contours[1])
print(x,y,w,h)
rasm2 = chizish[160:175+158,10:0+170]
plt.title("qora") 
plt.imshow(rasm2)


# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')
contours, boglama = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

chizish = np.copy(mushuk)
chizish = cv2.cvtColor(chizish,cv2.COLOR_BGR2RGB)
A= np.uint8
print(len(contours))
chizish = cv2.drawContours(chizish,contours,0,(0,255,0),10)
chizish = cv2.cvtColor(chizish,cv2.COLOR_BGR2RGB)
plt.title("A")
plt.imshow(chizish)


# In[24]:


plus_pixel = mushuk.reshape((-1,3))
plus_pixel = np.float32(plus_pixel)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.2)
k = 3
summa,label,center = cv2.kmeans(plus_pixel,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)

label = label.reshape((mushuk.shape[0],mushuk.shape[1]))
krasm = center[label.flatten()]
krasm = krasm.reshape((mushuk.shape))
plt.imshow(krasm)


# In[25]:


gray = cv2.cvtColor(mushuk,cv2.COLOR_RGB2GRAY)
_,binary = cv2.threshold(gray,254,255,cv2.THRESH_BINARY_INV)
erosion = cv2.erode(binary,kernel, -7)
plt.subplot(122)
plt.imshow(erosion,cmap='gray')


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
contours, boglama = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

chizish = np.copy(mushuk)
chizish = cv2.cvtColor(chizish,cv2.COLOR_BGR2RGB)
print(len(contours))
chizish = cv2.drawContours(chizish,contours,13,(0,255,0),10)
chizish = cv2.cvtColor(chizish,cv2.COLOR_BGR2RGB)
plt.subplot(121)
plt.title("A")
plt.imshow(chizish)
chizish1 = cv2.drawContours(chizish,contours,1,(0,255,0),10)
#chizish1 = cv2.cvtColor(chizish1,cv2.COLOR_BGR2RGB)
plt.title("B")
plt.subplot(122)
plt.imshow(chizish1)


# In[27]:


# harris algoritmi
shape = cv2.imread("wall.jpg")
shape = cv2.cvtColor(shape,cv2.COLOR_BGR2RGB)
plt.imshow(shape)
rasm_copy = np.copy(shape)


# In[28]:


gray = cv2.cvtColor(rasm_copy, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)

corner = cv2.cornerHarris(gray,2,3,0.04)

corner = cv2.dilate(corner,np.uint8(np.ones((13,13))))

plt.imshow(corner,cmap='gray')

