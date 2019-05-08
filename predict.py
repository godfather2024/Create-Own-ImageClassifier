#Importing libraries
#Dharit Sura 04/29/2019
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


parser = argparse.ArgumentParser(description='Predict.py')
parser.add_argument('--image_path', default='./flowers/test/5/image_05186.jpg', nargs='*', action="store", type = str)
parser.add_argument('--checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--json', dest="json", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store_true", dest="gpu")

args = parser.parse_args()
# Select parameters entered in command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.top_k:
    top_k = args.top_k
if args.json:
    filepath = args.json
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(filepath,'r') as f:
    cat_to_name = json.load(f)

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('Architecture not recognized')
    
    #model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    hidden_units = checkpoint['hidden_units']
    
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
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():        
        param.requires_grad = False
    
    return model

def process_image(image_path):
    
    size = 256,256
    crop_size = 224 
    #Resizing the image as per the requirement
    
    img_pil = Image.open(image_path)
    
    if img_pil.width > img_pil.height:
        img_pil.thumbnail((20000,256),Image.ANTIALIAS)
    else:
        img_pil.thumbnail((256,20000),Image.ANTIALIAS)
    
    left_margin = (size[0] - crop_size)/2
    top_margin = (size[1] - crop_size)/2
    right_margin = (left_margin + crop_size)
    bottom_margin = (top_margin + crop_size)
    
    image_crop = img_pil.crop((left_margin,top_margin,right_margin,bottom_margin))
    
    #Converting the values for the model between 0 and 1
    img_array = np.array(image_crop)
    np_image = img_array/255
    
    #Normalizing Images for the network
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    norm_image = (np_image - mean)/std
    pytorch_norm_image = norm_image.transpose(2,0,1)
    
    return pytorch_norm_image      

def predict(image_path, device, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file    
    model.to(device)
    img_torch = process_image(image_path)
    py_img_torch = torch.tensor(img_torch)
    py_img_torch = py_img_torch.float()
    
    py_img_torch = py_img_torch.unsqueeze(0)
    
    #new
    model.eval()
    softmax = model.forward(py_img_torch.cuda())
    pred = torch.exp(softmax)
    
	#Updated the topk to accept arguement passed value. 
    top_pred, top_labs = pred.topk(topk)
    top_pred = top_pred.detach().cpu().numpy().tolist()
    
    top_labs = top_labs.tolist()
    
    labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(cat_to_name)})
    labels = labels.set_index('class')
    
    
    labels = labels.iloc[top_labs[0]]
    labels['predictions'] = top_pred[0]
    
    #with torch.no_grad():
     #   output = model.forward(py_img_torch.cuda())
        
    #prob = F.softmax(output.data,dim=1)
    
    return labels

model = load_model(checkpoint)
print(model)
print('Model loaded successfully from the Checkpoint')
print('Starting to predict the model')
print(device)
labels = predict(image_path,device,model,top_k)
print(labels)
print('Completed!!!')
