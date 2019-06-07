import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import os
from PIL import Image

class EgoDexter(Dataset):
    '''Link to dataset: http://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm'''
    
    def __init__(self, config):
        self.root_dir = config['path']
        if config['scope'] == 'train':
            self.samples = get_splitted_image_names(self.root_dir)[0]
        else:
            self.samples = get_splitted_image_names(self.root_dir)[1]
         
        self.anno_2d_depth = get_2d_annotations(self.root_dir)
        self.anno_2d_depth = self.anno_2d_depth[self.anno_2d_depth['image_name'].isin(self.samples)]
        
        self.anno_3d = get_3d_annotations(self.root_dir)
        self.anno_3d = self.anno_3d[self.anno_3d['image_name'].isin(self.samples)]
        self.anno_3d = self.anno_3d.merge(right=self.anno_2d_depth[['image_name','not_nan']], 
                                          on='image_name', how='left')
        
        self.anno_3d = self.anno_3d[self.anno_3d['not_nan']==1]
        self.samples = np.array(self.anno_3d['image_name'])
        
        self.color_intrisics = np.array([[617.173, 0, 315.453],
                                         [0, 617.173, 242.259],
                                         [0, 0, 1]])
        self.color_extrisics = np.array([[1.0000, 0.00090442, -0.0074, 20.2365],
                                         [-0.00071933, 0.9997, 0.0248, 1.2846],
                                         [0.0075, -0.0248, 0.9997, 5.7360]])
        
        self.M_color = np.matmul(self.color_intrisics, self.color_extrisics)
        
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
        
        arr_anno_3d = self.anno_3d[self.anno_3d['image_name']==img_name] #add invisibility
        arr_anno_3d = np.reshape(np.array(arr_anno_3d)[:,:15], (3,5), order='F')
        arr_anno_3d = np.array(arr_anno_3d, dtype='float32')
        
        anno_3d_h = np.vstack((arr_anno_3d, np.ones((1,arr_anno_3d.shape[1]))))
        anno_2d_h = np.matmul(self.M_color, anno_3d_h)
        arr_anno_2d = anno_2d_h[:2,:] / anno_2d_h[2,:]
        
        arr_anno_2d[:,0] = arr_anno_2d[:,0] - 80
        arr_anno_2d = arr_anno_2d/480
        
        arr_anno_2d = np.array(arr_anno_2d, dtype='float32')
        
        arr_anno_3d = arr_anno_3d.flatten(order='F') ## add image transform
        arr_anno_2d = arr_anno_2d.flatten(order='F')

        
        
        return {'image': image, 
                'vector_2d': arr_anno_2d,
                'vector_3d': arr_anno_3d}

    
def get_splitted_image_names(root_dir, test_share=0.2):  #change split to test/train data
    img_names = np.array([])
    for folder in os.listdir(root_dir):
        images = os.listdir(root_dir + folder + '/color/')
        images = [folder + '/color/' + x for x in images]
        img_names = np.hstack((img_names, images))
            
    img_names = np.sort(img_names)
    np.random.shuffle(img_names)
    num_train = int(len(img_names) * (1-test_share))
    train = img_names[:num_train]
    test = img_names[num_train:]
    
    return train, test


def get_3d_annotations(root_dir):
    folders = os.listdir(root_dir)
    df_all = pd.DataFrame()
    for folder_name in folders:
        df = get_item_3d_annotations(root_dir, folder_name)
        df_all = pd.concat([df_all, df])

    df_all = df_all.reset_index(drop=True)
    for col in df_all.columns:
        df_all[col] = np.where(df_all[col]==-1, np.nan, df_all[col])
     
    return df_all


def get_item_3d_annotations(root_dir, folder_name):
    df = root_dir + folder_name + '/annotation.txt_3D.txt'
    df = pd.read_table(df, sep=';', header=None)
    df = df[df.columns[:-1]]
    df.columns = ['thumb','index','middle','ring','little']
    
    for col in df.columns:
        df[col +'_x'] = df[col].str.split(',', expand=True)[0]
        df[col +'_y'] = df[col].str.split(',', expand=True)[1]
        df[col +'_z'] = df[col].str.split(',', expand=True)[2]
        new_cols = [x for x in df.columns if x.find('_')>0]
    df = df[new_cols]
    
    for col in df.columns:
        df[col] = df[col].str.strip()
        df[col] = np.where(df[col]=='', np.nan, df[col])
        df[col] = df[col].astype(float)

    df['image_name'] = np.sort([folder_name  + '/color/' + x 
                                for x in os.listdir(root_dir + folder_name + '/color')])
    return df

def get_2d_annotations(root_dir):
    folders = os.listdir(root_dir)
    df_all = pd.DataFrame()
    for folder_name in folders:
        df = get_item_2d_annotations(root_dir, folder_name)
        df_all = pd.concat([df_all, df])

    df_all = df_all.reset_index(drop=True)
    for col in df_all.columns:
        df_all[col] = np.where(df_all[col]==-1, np.nan, df_all[col])
    
    df_all['not_nan'] = 1
    for col in df_all.columns[:10]:
        df_all['not_nan'] = df_all['not_nan'] * df_all[col].notnull()    
    return df_all


def get_item_2d_annotations(root_dir, folder_name):
    df = root_dir + folder_name + '/annotation.txt'
    df = pd.read_table(df, sep=';', header=None)
    df = df[df.columns[:-1]]
    df.columns = ['thumb','index','middle','ring','little']
    
    for col in df.columns:
        df[col +'_x'] = df[col].str.split(',', expand=True)[0]
        df[col +'_y'] = df[col].str.split(',', expand=True)[1]
        new_cols = [x for x in df.columns if x.find('_')>0]
    df = df[new_cols]
    
    for col in df.columns:
        df[col] = df[col].str.strip()
        df[col] = np.where(df[col]=='', np.nan, df[col])
        df[col] = df[col].astype(float)

    df['image_name'] = np.sort([folder_name  + '/color/' + x 
                                for x in os.listdir(root_dir + folder_name + '/color')])
    
    return df