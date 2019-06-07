import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat
import numpy as np
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from utils import GaussianBlur, get_heatmap_from_coordinates, TransformBlur, TransformRotate


class DexterObjectDataset(Dataset):
    '''Link to dataset: https://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/dexter+object.htm'''
    
    
    def __init__(self, config):
        
        '''keypoint order = thumb, index, middle, ring, little'''
        
        self.root_dir = config['path']
        self.augment = (bool(config['augment']) and (config['scope'] == 'train'))
        
        if config['scope'] == 'train':
            (self.samples, self.labels) = get_splitted_image_names(self.root_dir)['train']
        elif config['scope'] == 'val':
            (self.samples, self.labels) = get_splitted_image_names(self.root_dir)['val']
        else:
            print('Validation set is not available..')
            self.samples, self.labels3d = None, None
            
        
        self.bboxes = pd.read_csv('dataset/dexter_labeling/bbox_dexter+object.csv')
        
        color_intrisics = np.array([[587.45209, 0, 325],
                                    [0, 600.67456, 249],
                                    [0, 0, 1]])

        color_extrisics = np.array([[0.9999, 0.0034, 0.0161, 19.0473],
                                    [-0.0033, 1.0000, -0.0079, -1.8514], 
                                    [-0.0162, 0.0079, 0.9998, -4.7501]])
        
        self.M_color = np.matmul(color_intrisics, color_extrisics)
        
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
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
        
        self.transform_heatmap = transforms.Compose([
                                    GaussianBlur(),
                                    transforms.ToTensor()
                                    ])
        
        
        print('Images in Dexter+Object dataset: {}. Augmentation: {}'.format(
                                len(self.samples), self.augment))
        

    def __getitem__(self, idx):
        
        img_name = self.samples[idx]
        img_path = os.path.join(self.root_dir, 'data', img_name)
        image = Image.open(img_path)
        
        # keypoint order = thumb, index, middle, ring, little
        anno_3d = self.labels[idx]
        anno_3d = np.reshape(anno_3d, (5,3)).T
        anno_3d_h = np.vstack((anno_3d, np.ones((1,anno_3d.shape[1]))))
        anno_2d_h = np.matmul(self.M_color, anno_3d_h)
        anno_2d = anno_2d_h[:2,:] / anno_2d_h[2,:]
        anno_2d[0,:] = 640 - anno_2d[0,:] + 80 # strange corrections that make it work
        anno_2d[1,:] = 480 - anno_2d[1,:]
        anno_2d = anno_2d.T
        
        
        anno_2d, is_present = get_presence_ticket(anno_2d)
        
        bbox = np.array(self.bboxes[self.bboxes['img_name']==img_name][['x','y','w','h']], dtype='int')[0]
        
        
        if self.augment:
            image, anno_2d, bbox = rotate(image, anno_2d, bbox, is_present, self.augment)

        image, anno_2d, img_size = crop_hand(image, bbox, anno_2d, is_present)
        
        
        anno_2d = anno_2d / img_size
        heatmap = get_heatmap_from_coordinates(anno_2d, self.image_size, is_present)
        heatmap = self.transform_heatmap(heatmap)
        
        pr_ticket = (anno_2d > 1).sum(axis=1) + (anno_2d < 0).sum(axis=1)
        for i in range(len(is_present)):
            if pr_ticket[i] > 0:
                is_present[i] = 0
        
        anno_2d = anno_2d * np.reshape(is_present, (21,1))
        
        anno_2d = anno_2d.flatten() 
        anno_2d = torch.tensor(anno_2d, dtype=torch.float32)
        
        image = self.transform_image(image)
        

        return {'name': img_name,
                'image': image, 
                'heatmaps': heatmap,
                'vector_2d': anno_2d,
                'is_present': is_present,
                'img_size': img_size
               }

    def __len__(self):
        return len(self.samples)
    

    
def crop_hand(image, bbox, kpoints2d, is_present):
    
    x, y, w, h = 0, 1, 2, 3
    
    img_w, img_h = image.size
    
    kpoints2d_cropped = kpoints2d.copy()
    
    for i in range(len(is_present)):
        if is_present[i] == 1:
            kpoints2d_cropped[i,0] = kpoints2d[i,0] - bbox[x]
            kpoints2d_cropped[i,1] = kpoints2d[i,1] - bbox[y]
            
            if (kpoints2d_cropped[i,0]<0 or kpoints2d_cropped[i,1]<0):
                
                is_present[i] = 0
                kpoints2d_cropped[i,0] = 0
                kpoints2d_cropped[i,1] = 0

    cropped_image = image.crop((bbox[x], bbox[y], bbox[x]+bbox[w], bbox[y]+bbox[h]))
   
    return cropped_image, kpoints2d_cropped, bbox[w]
   
    
    
    
def get_presence_ticket(anno_2d):
    is_present_5 = [0]* 5
        
    for i in range(5):
        if 0 <= anno_2d[i,0] <= 640 and 0 <= anno_2d[i,1] <= 480:
            is_present_5[i] = 1
        
    is_present_21 = [0] * 21
    
    is_present_21[4] = is_present_5[0]
    is_present_21[8] = is_present_5[1]
    is_present_21[12] = is_present_5[2]
    is_present_21[16] = is_present_5[3]
    is_present_21[20] = is_present_5[4]
    
    
    anno_2d_new = np.zeros((21,2))
    
    anno_2d_new[4,:] = anno_2d[0,:] * is_present_21[4]
    anno_2d_new[8,:] = anno_2d[1,:] * is_present_21[8]
    anno_2d_new[12,:] = anno_2d[2,:] * is_present_21[12]
    anno_2d_new[16,:] = anno_2d[3,:] * is_present_21[16]
    anno_2d_new[20,:] = anno_2d[4,:] * is_present_21[20]
    
    is_present_21 = np.float32(is_present_21)
    
    
    return anno_2d_new, is_present_21



def get_splitted_image_names(path):

    folders = ['Grasp1','Grasp2', 'Occlusion','Rigid','Pinch','Rotate']
    #val_folders = ['Grasp2']
    val_folders = ['Rotate']
    
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    
    for fd in folders:
        
        fd_path = os.path.join(path, 'data', fd, 'color')
        files = os.listdir(fd_path)
        files = [os.path.join(fd, 'color', x) for x in files]
        files = np.sort(files)
        
        fn_anno_3d = os.path.join(path, 'data', fd, 'annotations/', fd + '3D.txt')
        df_anno_3d = pd.read_table(fn_anno_3d, sep=';', header=None)
        cols = [0,1,2,3,4]
        df_anno_3d = df_anno_3d[cols]

        for col in cols:
            new_cols = df_anno_3d[col].str.replace(' ', '').str.split(',', expand=True)
            df_anno_3d[[str(col)+'_x', str(col)+'_y', str(col)+'_z']] = new_cols
    
        df_anno_3d = df_anno_3d[df_anno_3d.columns[5:]]
        df_anno_3d = np.array(df_anno_3d, dtype='float32')
        
    
        if fd in val_folders:
            val_images.extend(files)
            val_labels.extend(df_anno_3d)
        else: 
            train_images.extend(files)
            train_labels.extend(df_anno_3d)
    
    
    return {'train': (train_images, train_labels),
            'val': (val_images, val_labels)}



def rotate(image, anno_2d, bbox, is_present, augment):
    angle_low = -180
    angle_high = 180
    angle = np.random.uniform(low=angle_low, high=angle_high)
    
    #rotate image 
    img_new = image.rotate(angle)
    
    
    #stack bbox and kpoints
    anno_2d = anno_2d.T
    bbox_xy = np.array([[bbox[0], bbox[0], bbox[0]+ bbox[2], bbox[0]+ bbox[2]],
                        [bbox[1], bbox[1]+ bbox[3], bbox[1], bbox[1]+ bbox[3]]])
    
    kpoints = np.hstack((anno_2d, bbox_xy))
    
    #rotate bbox and keypoints
    angle = -np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    
    w,h = image.size
    
    kpoints[0,:] = kpoints[0,:] - w/2
    kpoints[1,:] = kpoints[1,:] - h/2
    kpoints_new = np.matmul(rotation_matrix, kpoints)
    kpoints_new[0,:] = kpoints_new[0,:] + w/2
    kpoints_new[1,:] = kpoints_new[1,:] + h/2
    
    
    bbox_xy_new = kpoints_new[:,21:]
    kpoints_new = kpoints_new[:,:21]
    kpoints_new = kpoints_new.T
    
    kpoints_new = kpoints_new * np.reshape(is_present, (21,1))
    
    #create new bbox
    min_x = int(bbox_xy_new[0:].min())
    min_y = int(bbox_xy_new[1:].min())
    max_x = int(bbox_xy_new[0:].max())
    max_y = int(bbox_xy_new[1:].max())
    
    size = max(max_x-min_x, max_y-min_y)
    
    if augment:
        scale = np.random.uniform(low=0.70, high=1)
        center_x = min_x + size//2
        center_y = min_y + size//2
        
        new_size = int(size * scale)
        min_x = center_x - new_size // 2
        min_y = center_y - new_size // 2
        size = new_size
        
    bbox_new = [min_x, min_y, size, size]
    
    return img_new, kpoints_new, bbox_new