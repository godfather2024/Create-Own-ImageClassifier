#Importing libraries
#Dharit Sura 04/28/2019
# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as matplt
import seaborn as sb

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
import json
from matplotlib.ticker import FormatStrFormatter

from collections import OrderedDict
from os import listdir
import time
import copy
import argparse

parser = argparse.ArgumentParser(description='Train.py')
#Defaults if not input
#set to 15 for high accuracy. #Test run with 1

parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.0001)
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=15)
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=5120)
parser.add_argument('--gpu',action='store_true',default="gpu")

args = parser.parse_args()
#If values provided 
# Select parameters entered in command line
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#Creating model
def create_model(arch='vgg16',hidden_units=5120,learning_rate=0.0001):
    model =  getattr(models,arch)(pretrained=True)
    in_features = model.classifier[0].in_features
    
    #Freeze feature parameters so as not to backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
    
    #Classifier for model
    classifier = nn.Sequential(OrderedDict([
                           ('fc1',nn.Linear(in_features,hidden_units)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.15)),
                           ('fc2',nn.Linear(hidden_units,512)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.15)),
                           ('fc3',nn.Linear(512,102)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learning_rate)
    return model, criterion, optimizer

model, criterion, optimizer = create_model(arch, hidden_units, learning_rate)
print("Model built Successfully!!!")

#Loading Data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

#Validation Transforms
validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
#TestTransform
test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
validation_imageDataSet = datasets.ImageFolder(valid_dir,transform=validation_transforms)
test_imageDataSet = datasets.ImageFolder(test_dir,transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
validation_dataLoaders = torch.utils.data.DataLoader(validation_imageDataSet, batch_size=64, shuffle=True)
test_dataLoaders = torch.utils.data.DataLoader(test_imageDataSet, batch_size=64, shuffle=True)

print("DataLoaded")
#Training the model 
#for test run using lower epoch 
def train_model(model, criterion, optimizer, device, epochs= epochs, dataset= dataloaders):
    print('Number of epochs ' + str(epochs))
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 5
    print("Training Started")
    for epoch in range(epochs):
        for inputs, labels in dataloaders:

            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            optimizer.zero_grad()

            # Forward and backward passes
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validation_dataLoaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(validation_dataLoaders):.3f}.. "
                      f"Test accuracy: {accuracy/len(validation_dataLoaders):.3f}")
                print(
                    f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
                running_loss = 0
                model.train()
    print("Training Complete")
    return model

model_trained = train_model(model, criterion, optimizer, device, epochs, dataloaders)
print("Training Complete Save the model")

def save_model(model_trained):
    model_trained.class_to_idx = image_datasets.class_to_idx
    model_trained.to(device)
    print("Saving model checkpoint")
    checkpoint = {              
             'state_dict': model_trained.state_dict(),
             #'classifier': classifier,
             'batch_size': 64,
             'epochs': epochs,
             'arch': arch,
             'hidden_units' : hidden_units,
             'optimizer': optimizer.state_dict(),
             'class_to_idx': model_trained.class_to_idx,
                 }
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir)
    print("Model Saved")
    
save_model(model_trained)
print(model_trained)
print('Your model has been successfully saved.')