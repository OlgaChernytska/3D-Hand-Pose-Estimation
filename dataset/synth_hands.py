import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import cv2
import random

from utils import GaussianBlur, get_heatmap_from_coordinates, TransformScale, TransformRotate, TransformBlur

''' Normalization rule for 3D locations: 
Middle finger MCP joint is at the origin. 
The values are normalized such that the length between middle finger MCP and wrist is 1.''' 


class SynthHandsDataset(Dataset):
    def __init__(self, config):
        
        self.root_dir = config['path']
        self.root_dir_background = config['path_background']
        self.augment = (bool(config['augment']) and (config['scope'] == 'train'))
        #self.augment = True
        
        if config['scope'] == 'train':
            self.samples = get_splitted_image_names(self.root_dir)['train']
            self.backgrounds = get_background_images(self.root_dir_background)['train']
        elif config['scope'] == 'val':
            self.samples = get_splitted_image_names(self.root_dir)['val']
            self.backgrounds = get_background_images(self.root_dir_background)['val']
        else:
            self.samples = get_splitted_image_names(self.root_dir)['test']
            self.backgrounds = get_background_images(self.root_dir_background)['test']
            
        ##===================
        ##Used only for model valuation - images with objects and without objects
        self.with_objects = None
        if self.with_objects == True:
            self.samples = [x for x in self.samples if x.find('_object')>=0]
        if self.with_objects == False:
            self.samples = [x for x in self.samples if x.find('_noobject')>=0]
        ##===================
        
        self.image_size = 128
        
        if self.augment:
            self.transform_image = transforms.Compose([
                           transforms.CenterCrop((480,480)), 
                           transforms.Resize((self.image_size, self.image_size)), 
                           transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25),
                           TransformBlur(),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])
            
            self.transform_scale = TransformScale()
            self.transform_rotatate = TransformRotate()
        
        else: 
            self.transform_image = transforms.Compose([
                           transforms.CenterCrop((480,480)), 
                           transforms.Resize((self.image_size, self.image_size)), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                           ])
        
        self.transform_heatmap = transforms.Compose([
                                    GaussianBlur(),
                                    transforms.ToTensor()
                                    ])
        
        # projecting 3D coordintated onto color image plane
        self.color_intrisics_matrix = np.array([[617.173, 0, 315.453],
                                                [0, 617.173, 242.259],
                                                [0, 0, 1]])

        self.color_extrisics_matrix = np.array([[1, 0, 0, 24.7],
                                                [0, 1, 0, -0.0471401],
                                                [0, 0, 1, 3.72045]])

        print('Images in SynHands dataset: {}. Augmentation: {}'.format(len(self.samples), self.augment))
        
    def __getitem__(self, idx):
       
        img_name = self.samples[idx]
        image = Image.open(img_name)
        w,h = image.size
        
        fn_anno_3d = img_name.replace('color.png','joint_pos.txt')
        anno_3d = np.loadtxt(fn_anno_3d, delimiter=',')
        anno_3d = np.reshape(anno_3d, (21,3))
        
        # normalized 3D pose is the same whether performed either on 3d_depth or 3d_color
        anno_3d_n = anno_3d - anno_3d[9,:]
        anno_3d_n = anno_3d_n / np.sqrt(np.sum(anno_3d_n[0,:] ** 2))
        anno_3d_color = np.matmul(self.color_extrisics_matrix, 
                                  np.hstack((anno_3d, np.ones((anno_3d.shape[0],1)))).T)
        
        if self.augment:
            image, anno_3d_color = self.transform_rotatate(image, anno_3d_color)
             
        anno_2d = np.matmul(self.color_intrisics_matrix, anno_3d_color).T
        anno_2d = anno_2d[:,:2]/np.reshape(anno_2d[:,2], (anno_2d.shape[0],1))
        
        if self.augment:
            image, anno_2d = self.transform_scale(image, anno_2d)
        
        anno_2d[:,0] = (anno_2d[:,0] - ((w-h)/2)) 
        anno_2d = anno_2d / h
        
        image = np.array(image)[:,:,:3]
        image = background_augmentation(image, self.backgrounds)
        image = Image.fromarray(image)
        image = self.transform_image(image)
        
        is_present = np.array([1] * 21)
        heatmap = get_heatmap_from_coordinates(anno_2d, self.image_size, is_present)
        heatmap = self.transform_heatmap(heatmap)
        
        anno_3d_tree = tree_from_vector(anno_3d_n)
        anno_3d_tree = anno_3d_tree.flatten()
        anno_3d_tree = torch.tensor(anno_3d_tree, dtype=torch.float32)
        
        anno_3d_n = anno_3d_n.flatten()
        anno_3d_n = torch.tensor(anno_3d_n, dtype=torch.float32)
        
        anno_2d = anno_2d.flatten()
        anno_2d = torch.tensor(anno_2d, dtype=torch.float32)
        
        
        
        return {'name': img_name,
                'image': image, 
                'heatmaps': heatmap,
                'vector_2d': anno_2d,
                'vector_3d': anno_3d_n,
                'vector_3d_tree': anno_3d_tree,
                'is_present': is_present}
    

    def __len__(self):
        return len(self.samples)



def get_splitted_image_names(root_dir): 
    image_names = np.array([])
    data_classes = [x for x in os.listdir(root_dir) if x.find('.')<0]
    for data_class in data_classes:
        seqs = os.listdir(root_dir + '/' + data_class)
        seqs = [data_class + '/' + x for x in seqs]
        for seq in seqs:
            cams = os.listdir(root_dir + '/' + seq)
            cams = [seq + '/' + x for x in cams] 
            for cam in cams:
                folders = os.listdir(root_dir + '/' + cam)
                folders = [cam + '/' + x for x in folders]
                for folder in folders:
                    files = os.listdir(root_dir + '/' + folder)
                    files = [root_dir + folder + '/' + x for x in files if x.find('_color.png')>0]
                    image_names = np.hstack((image_names,files))
    
    image_names = np.sort(image_names)
    np.random.seed(42)
    np.random.shuffle(image_names)
    val = image_names[:5000]
    test = image_names[5000:10000]
    train = image_names[10000:]
    
    return {'train': train, 'test': test, 'val': val}


def get_background_images(root_dir):
    img_names = np.array([])
    for path, subdirs, files in os.walk(root_dir): 
        files = [os.path.join(path,x) for x in files if x.find('.jpg')>=0]
        img_names = np.hstack((img_names, files))
        
    img_names = np.sort(img_names)
    np.random.seed(42)
    np.random.shuffle(img_names)
    
    val = img_names[:2000]
    test = img_names[2000:4000]
    train = img_names[4000:]
    
    return {'train': train, 'test': test, 'val': val}


def background_augmentation(image, backgrounds_arr):
    
    mask_green = np.array(image[:,:,0]==14, dtype='int') * np.array(image[:,:,1]==255, dtype='int') \
                   * np.array(image[:,:,2]==14, dtype='int')
    mask_white = np.array(image[:,:,0]==0, dtype='int') * np.array(image[:,:,1]==0, dtype='int') \
                   * np.array(image[:,:,2]==0, dtype='int')
    mask = mask_green + mask_white
   
    b_idx = random.choice(range(len(backgrounds_arr)))
    b_name = backgrounds_arr[b_idx]
    b_image = Image.open(b_name)
    
    w, h = b_image.size
    if w<=640 or h<=480:
        b_image = b_image.resize((640,480))
    else:
        x = random.choice(range(w-640+1))
        y = random.choice(range(h-480+1))
        b_image = b_image.crop((x,y,x+640,y+480))
    b_image = np.array(b_image)
        
    if len(b_image.shape)==3:
        image = b_image * mask[:,:,None] + image * (1 - mask[:,:,None])
    else:
        image = b_image[:,:,None] * mask[:,:,None] + image * (1 - mask[:,:,None])
    
    image = np.array(image, dtype='uint8')
    return image


    
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
  

