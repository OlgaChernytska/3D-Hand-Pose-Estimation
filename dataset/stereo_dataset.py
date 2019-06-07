import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.io import loadmat
import numpy as np
import cv2
import os
import numpy as np
from PIL import Image
from utils import GaussianBlur, get_heatmap_from_coordinates, TransformBlur, TransformRotate


class StereoDataset(Dataset):
    '''
    Links to dataset: https://sites.google.com/site/zhjw1988/,
    https://www.dropbox.com/sh/ve1yoar9fwrusz0/AAAfu7Fo4NqUB7Dn9AiN8pCca?dl=0
    '''
    
    def __init__(self, config):
        
        self.root_dir = config['path']
        self.augment = (bool(config['augment']) and (config['scope'] == 'train'))
        
        if config['scope'] == 'train':
            (self.samples, self.labels3d) = get_splitted_image_names(self.root_dir)['train']
        elif config['scope'] == 'val':
            (self.samples, self.labels3d) = get_splitted_image_names(self.root_dir)['val']
        else:
            print('Validation set is not available..')
            self.samples, self.labels3d = None, None
            
            
        self.image_size = 128

        self.intrinsic = np.array([[607.92271, 0, 314.78337],
                                   [0, 607.88192, 236.42484],
                                   [0, 0, 1]
                                  ])
        if self.augment:
            self.transform_image = transforms.Compose([
                               transforms.Resize((self.image_size, self.image_size)), 
                               transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.25),
                               TransformBlur(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
            
            
            self.transform_rotatate = TransformRotate(rotation_type='2d')
            
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
        
        
        
        
        
        print('Images in Stereo dataset: {}. Augmentation: {}'.format(len(self.samples),self.augment))
        
        

    def __getitem__(self, idx):
        
    
        kpoints3d = self.labels3d[idx]
        kpoints2d = (self.intrinsic @ kpoints3d.T).T
        kpoints2d = kpoints2d[:, :2] / kpoints2d[:, 2:]
        
        init = kpoints2d
        img_name = self.samples[idx]
        image = Image.open(img_name)
        
        if self.augment:
            image, kpoints2d = self.transform_rotatate(image, kpoints2d.T)
            kpoints2d = kpoints2d.T
              
            
        image, kpoints2d, img_size = crop_hand(image, kpoints2d, self.augment)
        
        is_present = np.float32([0] + [1] * 20)
        kpoints2d = kpoints2d * np.reshape(is_present, (21,1))
        
        kpoints2d = kpoints2d / img_size
        
        heatmap = get_heatmap_from_coordinates(kpoints2d, self.image_size, is_present)
        heatmap = self.transform_heatmap(heatmap)
        
        
        kpoints2d = kpoints2d.flatten()
        kpoints2d = torch.tensor(kpoints2d, dtype=torch.float32)
        
        image = self.transform_image(image)
        

        return {'name': img_name,
                'image': image, 
                'heatmaps': heatmap,
                'vector_2d': kpoints2d,
                'is_present': is_present,
                'img_size': img_size
               }

    def __len__(self):
        return len(self.samples)

    
    
def get_splitted_image_names(path): 
    
    # rotation/translation vectors should be with negative sign 
    # due to backward transformation from depth2color camera space
    r_vec = -np.array([[0.00531, -0.01196, 0.00301]])
    t_vec = -np.array([-24.0381, -0.4563, -1.2326])
    r_mat, _ = cv2.Rodrigues(r_vec)
    transform_matrix = np.hstack([r_mat, t_vec.reshape((3, 1))])

    images = []
    labels = []
    
    finge_index = [0,17,18,19,20,13,14,15,16,9,10,11,12,5,6,7,8,1,2,4,3]
    
    for chunk in range(1, 7):
        for scope in ['Random', 'Counting']:
            folder_name = 'images/B{}{}'.format(chunk,scope)
            mat = loadmat(os.path.join(path, 'labels', 'B{}{}_SK.mat'.format(chunk,scope)))
            kpoints3d = mat['handPara']

            for i in range(1500):
                img = os.path.join(path, folder_name, 'SK_color_{}.png'.format(i))
                images.append(img)
                kps3d = kpoints3d[..., i].T
                kps3d_h = np.hstack([kps3d, np.ones((21, 1))])
                kps3d = (transform_matrix @ kps3d_h.T).T
                kps3d = kps3d[finge_index]
                labels.append(kps3d)
            
    
    val_images = images[:3000]
    val_labels = labels[:3000]
    
    train_images = images[3000:]
    train_labels = labels[3000:]
    
    return {'train': (train_images, np.array(train_labels)), 
            'val': (val_images, np.array(val_labels))}


def crop_hand(image, kpoints2d, augment):

    w,h = image.size
    final_size = None
    
    if augment:
        boundary = np.random.uniform(low=40, high=100)
    else:
        boundary = 40
    
    
    max_x = int(min(np.max(kpoints2d[:,0])+boundary,w))
    min_x = int(max(np.min(kpoints2d[:,0])-boundary,0))
    
    max_y = int(min(np.max(kpoints2d[:,1])+boundary,h))
    min_y = int(max(np.min(kpoints2d[:,1])-boundary,0))
    
    crop_size_x = max_x-min_x
    crop_size_y = max_y-min_y
   
    
    if not crop_size_x % 2 == 0:
        max_x -= 1
        crop_size_x = max_x-min_x
        
    if not crop_size_y % 2 == 0:
        max_y -= 1
        crop_size_y = max_y-min_y
    
    if crop_size_y > crop_size_x:
        half_diff = (crop_size_y - crop_size_x) // 2
    
        min_x -= half_diff
        max_x += half_diff
        final_size = crop_size_y
        
    else:
        half_diff = (crop_size_x - crop_size_y) // 2
        min_y -= half_diff
        max_y += half_diff
        final_size = crop_size_x
        
        
    cropped_image = image.crop((min_x,min_y,max_x,max_y))
    
    #print((max_x - min_x) == (max_y - min_y))
    
    kpoints2d_cropped = kpoints2d.copy()
    kpoints2d_cropped[:,0] = kpoints2d_cropped[:,0] - min_x
    kpoints2d_cropped[:,1] = kpoints2d_cropped[:,1] - min_y
    
    return cropped_image, kpoints2d_cropped, final_size