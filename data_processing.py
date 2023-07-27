import numpy as np
import torchvision
import torchvision.transforms as transforms
import cv2
import torch

def load_data():
    trainset = torchvision.datasets.MNIST(root='/pscratch/sd/g/gchen4/output_VAE_MNIST/data', train = True, download = True, transform = transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='/pscratch/sd/g/gchen4/output_VAE_MNIST/data', train = False, download = True, transform = transforms.ToTensor())
    return trainset, testset

def process_data(trainset, testset):
    xtrain = trainset.data.numpy()
    ytrain = trainset.targets.numpy()
    x_val_pre = testset.data[:1000].numpy()
    y_val = testset.targets[:1000].numpy()

    #Creating x_train and y_train with 1000 images from each class and binarizing the pixels
    count = np.zeros(10)
    idx = []
    for i in range(0, len(ytrain)):
      for j in range(10):
        if(ytrain[i] == j):
          count[j] += 1
          if(count[j]<=1000):
            idx = np.append(idx, i)

    y_train = ytrain[idx.astype('int')]
    x_train_pre = xtrain[idx.astype('int')]

    #Resizing the images from 28x28 to 14x14
    r,_,_ = x_train_pre.shape
    x_train = np.zeros([r,14,14])
    for i in range(r):
      a = cv2.resize(x_train_pre[i].astype('float32'), (14,14)) # Resizing the image from 28*28 to 14*14
      x_train[i] = a

    r,_,_ = x_val_pre.shape
    x_val = np.zeros([r,14,14])
    for i in range(r):
      a = cv2.resize(x_val_pre[i].astype('float32'), (14,14)) # Resizing the image from 28*28 to 14*14
      x_val[i] = a

    # Binarizing
    x_train = np.where(x_train > 128, 1, 0)
    x_val = np.where(x_val > 128, 1, 0)
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)

    return x_train, y_train, x_val, y_val

def create_dataloaders(x_train, y_train, x_val, y_val, batch_size=32):
    trainloader = torch.utils.data.DataLoader([[x_train[i], y_train[i]] for i in range(len(y_train))], shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader([[x_val[i], y_val[i]] for i in range(len(y_val))], shuffle=True, batch_size=100)
    return trainloader, testloader
