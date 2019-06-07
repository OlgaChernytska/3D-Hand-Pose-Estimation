import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image

class EgoDexter(Dataset):
    def __init__(self, config):
        self.root_dir = config['path']
        
        self.samples = get_image_names(self.root_dir)[:100]
       
        self.transform_image = transforms.Compose([
                           transforms.CenterCrop((480,480)), 
                           transforms.Resize((128,128)), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        print('Images in EgoDexter dataset: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        img_name = self.samples[idx]
        image = Image.open(self.root_dir + img_name)
        image = self.transform_image(image)
      

        return {'image': image}

    
def get_image_names(root_dir): 
    #folder = 'Fruits'
    #img_names = os.listdir(root_dir + folder + '/color/')
    #img_names = [folder + '/color/' + x for x in img_names]       
    #img_names = np.sort(img_names)
    
    
    
    img_names = np.array([])
    for folder in os.listdir(root_dir):
        images = os.listdir(root_dir + folder + '/color/')
        images = [folder + '/color/' + x for x in images]
        img_names = np.hstack((img_names, images))
            
    img_names = np.sort(img_names)
    return img_names


