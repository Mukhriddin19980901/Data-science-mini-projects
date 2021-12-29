#!/usr/bin/env python
# coding: utf-8

# In[26]:


# kerakli kutubxonalarni yuklab olamiz
# Libraries that we use
from torch import nn,optim,cuda,from_numpy
from torch.utils import data
from torchvision import datasets,transforms
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np 


# cuda ishlayaptimi tekshiramiz
device = 'cuda' if cuda.is_available() else 'cpu'
print(device) 
af = ''
# Datasetlarni hammasini yuklab olamiz (10 ta)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Bizga tensorlar hajmini normallashtirb beradi.  

train_dataset = datasets.CIFAR10(root='C:/Users/USER/datasets2021/CIFAR10/',
                              train = True,
                              transform = transform,
                              download = True)
test_dataset = datasets.CIFAR10(root='C:/Users/USER/datasets2021/CIFAR10/',
                              train = False,
                              transform=transform)
batch_size = 4
learning_rate = 0.001
 
# Yuklab olingan malumotni training va test uchun kerakli holatga keltiramiz qayta nomlaymiz
train_loader = data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
classes = ("samolyot","avtoulov",'qush','mushuk','kiyik','it','qurbaqa','ot','kema','yuk mashinasi')

# training modelimizni yozamiz
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.con_lay1 = nn.Conv2d(3,6,5) # convolution layer-(1)  kiruvchi kanal 3 chunki rasmda 3 ta kanal bor RGB,6 chiquvchi kanal filterlar(height,width,5 bu kernel
        self.pool = nn.MaxPool2d(2, 2)  # pooling qatlam pixelini kichiklashtirish orqali o'lchamni kichik qiladi
        self.con_lay2 = nn.Conv2d(6,16,5) # outputdan chiqqan 6 ni 16 ta filterdan otkazadi
        self.linear1 = nn.Linear(16*5*5,120) 
        self.linear2 = nn.Linear(120,84)
        self.linear3 = nn.Linear(84,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.pool(self.relu(self.con_lay1(x)))
        x = self.pool(self.relu(self.con_lay2(x)))
        x = x.view(-1,16*5*5)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
model = Model()
model.to(device)
criteria = nn.CrossEntropyLoss()
optimize = optim.SGD(model.parameters(), lr = 0.001)
epochs = 4
for epoch in range(epochs):
    for batch_index,datalar in enumerate(train_loader):
        x,label = datalar[0].to(device), datalar[1].to(device)
        optimize.zero_grad()
        predict = model(x)
        loss = criteria(predict,label)
        loss.backward()
        optimize.step()
        
    print(f"Epoch - {epoch+1} loss  : {loss.item()} %")
    print("...")
print("Training finished  ")

with torch.no_grad():
    n_togri = 0
    n_namuna = 0
    n_class_togri = [0 for i in range(10)]
    n_class_namuna = [0 for i in range(10)]
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _,predicted =torch.max(output,1)
        n_namuna = labels.size(0)
        n_togri = (predicted == labels).sum().item()
        
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_togri[label]+=1
            n_class_namuna[label]+=1
            
    aniq = 100.0*n_togri / n_namuna
    print(f"Testning aniqlik darajasi {aniq} % ")
        
    for i in range(10):
        aniq = 100.0*n_class_togri[i]/n_class_namuna[i]
        af=classes[i]
        print(f"{af} classining aniqlik darajasi {aniq} % ")


# In[37]:


get_ipython().run_line_magic('matplotlib', 'qt')
def imshow(img):
    img = img/2*0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
dataiter = iter(train_loader)
images,labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))


# In[ ]:





# In[ ]:




