import torch
import cv2
import os
import numpy as np
from torchvision import datasets, transforms, models
from model import Net

def get_model(path):
    model=Net()
    state_dict=torch.load(path,map_location='cpu')
    model.load_state_dict(state_dict)
    return model

def get_transform():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(480),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        # transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
    return transform

def get_PIL_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(480),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        # transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ])
    return transform

def image_to_tensor(path,transform):
    img=cv2.imread(os.getcwd()+path)
    img_torch=torch.from_numpy(img)
    img_torch=transform(img_torch)
    print(img_torch.shape)
    return img_torch


def get_result(image,model):
    if torch.cuda.is_available():
        model.cuda()
        image.cuda()
    else:
        model.to('cpu')
    model.eval()
    probability=model.forward(image)
    probability=torch.exp(probability).detach().numpy()
    if(probability[0][0]>probability[0][1]):
        return "NORMAL"
    else:
        return "Pnemonia"


