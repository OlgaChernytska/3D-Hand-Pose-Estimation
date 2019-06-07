import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import cv2

from utils import GaussianBlur, get_heatmap_from_coordinates, TransformBlur

''' Normalization rule for 3D locations: 
Middle finger MCP joint is at the origin. 
The values are normalized such that the length between middle finger MCP and wrist is 1.''' 

class GANeratedDataset(Dataset):
    '''Link to dataset: https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/GANeratedDataset.htm'''
    
    def __init__(self, config):
        
        self.root_dir = config['path']
        self.augment = (bool(config['augment']) and (config['scope'] == 'train'))
        #self.augment = True
        
        if config['scope'] == 'train':
            self.samples = get_splitted_image_names(self.root_dir)['train']
        elif config['scope'] == 'val':
            self.samples = get_splitted_image_names(self.root_dir)['val']
        else:
            self.samples = get_splitted_image_names(self.root_dir)['test']
            
        ##===================
        ##Used only for model valuation - images with objects and without objects
        self.with_objects = None
        if self.with_objects == True:
            self.samples = [x for x in self.samples if x.find('/withObject/')>=0]
        if self.with_objects == False:
            self.samples = [x for x in self.samples if x.find('/noObject/')>=0]
        ##===================
        
        self.image_size = 128
        
        if self.augment: 
            self.transform_image = transforms.Compose([
                           transforms.Resize((self.image_size, self.image_size)), 
                           transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25),
                           TransformBlur(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])
            
        else:
            self.transform_image = transforms.Compose([
                           transforms.Resize((self.image_size, self.image_size)), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.transform_heatmap = transforms.Compose([
                                    GaussianBlur(),
                                    transforms.ToTensor()
                                    ])

    
        print('Images in GANerated dataset: {}. Augmentation: {}'.format(len(self.samples), self.augment))
        
    def __getitem__(self, idx):
        img_name = self.samples[idx]
        image =  Image.open(img_name)
        (img_size,_) = image.size
       
        fn_2d_keypoints = img_name.replace('color_composed.png', 'joint2D.txt')
        arr_2d_keypoints = np.loadtxt(fn_2d_keypoints, delimiter=',')
        arr_2d_keypoints = arr_2d_keypoints / img_size
        
        fn_3d_keypoints = img_name.replace('color_composed.png', 'joint_pos.txt')  
        arr_3d_keypoints = np.loadtxt(fn_3d_keypoints, delimiter=',')
        
        is_present = np.array([1] * 21)
        heatmap = get_heatmap_from_coordinates(np.reshape(arr_2d_keypoints, (21,2)), 
                                               self.image_size, is_present)
        heatmap = self.transform_heatmap(heatmap)
        
        
        anno_3d_tree = tree_from_vector(np.reshape(arr_3d_keypoints, (21,3)))
        anno_3d_tree = anno_3d_tree.flatten()
        anno_3d_tree = torch.tensor(anno_3d_tree, dtype=torch.float32)
        
        arr_2d_keypoints = torch.tensor(arr_2d_keypoints, dtype=torch.float32)
        arr_3d_keypoints = torch.tensor(arr_3d_keypoints, dtype=torch.float32)
    
        image = self.transform_image(image)
        
        
        
        return {'name': img_name,
                'image': image, 
                'heatmaps': heatmap, 
                'vector_2d': arr_2d_keypoints,
                'vector_3d': arr_3d_keypoints,
                'vector_3d_tree': anno_3d_tree,
                'is_present': is_present}

    def __len__(self):
        return len(self.samples)



def get_splitted_image_names(root_dir): 
    img_names = np.array([])
    
    for img_type in ['noObject/','withObject/']:
        folders = os.listdir(root_dir + img_type)
        folders = [img_type + x + '/' for x in folders if len(x)==4]
        
        for folder in folders:
            images = os.listdir(root_dir + folder)
            images = [root_dir + folder + x for x in images if x.find('.png')>0]
            img_names = np.hstack((img_names, images))
            
    img_names = np.sort(img_names)
    np.random.seed(42)
    np.random.shuffle(img_names)
    
    val = img_names[:5000]
    test = img_names[5000:10000]
    train = img_names[10000:]
    
    return {'train': train, 'test':test, 'val': val}


    
def tree_from_vector(anno_3d):
    new_vectors_ids = [(1,2),(2,3),(3,4),(4,5),
                   (1,6),(6,7),(7,8),(8,9),
                   (11,12),(12,13),
                   (1,14),(14,15),(15,16),(16,17),
                   (1,18),(18,19),(19,20),(20,21)]
    
    for (s,e) in new_vectors_ids:
        new_vector = anno_3d[e] - anno_3d[s]
        anno_3d = np.vstack((anno_3d, new_vector))
    
    return anno_3d
  
