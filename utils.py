import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import ImageFilter


KPOINTS_COUNT = 21


def tensor_pt(array):
    return cuda(torch.Tensor(array)).float()


def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

    

def latent2coords(latent_heatmaps):
    heatmaps2d = latent_heatmaps[:, :KPOINTS_COUNT]
    origin_shape = heatmaps2d.size()
    flatten_heatmaps = heatmaps2d.view((heatmaps2d.size(0), heatmaps2d.size(1), -1))
    normalized_hmaps = F.softmax(flatten_heatmaps, dim=-1)
    normalized_hmaps = normalized_hmaps.view(origin_shape)

    depth_heatmaps = latent_heatmaps[:, KPOINTS_COUNT:]
    z = (normalized_hmaps * depth_heatmaps).sum(-1).sum(-1)
    h, w = latent_heatmaps.size(2), latent_heatmaps.size(3)
    hor = torch.arange(w).repeat(1, h).float().view(h, w).unsqueeze(0).unsqueeze(0).cuda().detach()
    vert = torch.arange(h).repeat(w, 1).float().view(w, h).transpose(1, 0).unsqueeze(0).unsqueeze(0).cuda().detach()
    x = (hor * normalized_hmaps).sum(-1).sum(-1)
    y = (vert * normalized_hmaps).sum(-1).sum(-1)
    return torch.stack([x, y, z], -1)


# ============= PREPROCESSING ==============

class GaussianBlur(object):
    def __init__(self, kernel_size=(61,61), sigma=3):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        sample = np.array(sample) * 255.
        h,w,k = sample.shape
        for i in range(k):
            heatmap = sample[:,:,i]
            if heatmap.max()!=0:
                blurred = cv2.GaussianBlur(heatmap, self.kernel_size, self.sigma)
                blurred = np.array(blurred/blurred.max())
                sample[:,:,i] = blurred
        
        return np.float32(sample) 
    
    

class TransformBlur(object):
    def __init__(self, low=0, high=0.8):
        self.low = low
        self.high = high

    def __call__(self, image):
        radius = np.random.uniform(low=self.low, high=self.high)
        image = image.filter(ImageFilter.GaussianBlur(radius))
        return image  
    
    
class TransformScale(object):
    def __init__(self, scale_percent_low=-0.25, scale_percent_high=0.15):
        self.scale_percent_low = scale_percent_low
        self.scale_percent_high = scale_percent_high
      
    def __call__(self, img, kpoints2d):
        
        percent = np.random.uniform(low=self.scale_percent_low, high=self.scale_percent_high)
        
        #changing image scale 
        w,h = img.size
        add_w = int(w * percent)
        add_h = int(h * percent)
        img_new = img.resize((w+add_w, h+ add_h))
        img_new = img_new.crop((add_w//2, add_h//2, w+add_w//2, h+add_h//2))
    
        #changing 2d coordinates
        kpoints2d_new = np.ones_like(kpoints2d)
        kpoints2d_new[:,0] = kpoints2d[:,0] * (1+percent) - add_w//2
        kpoints2d_new[:,1] = kpoints2d[:,1] * (1+percent) - add_h//2
    
        return img_new, kpoints2d_new
    
    
class TransformRotate(object):
    def __init__(self, rotation_type='3d', angle_low=-180, angle_high=180):
        self.angle_low = angle_low
        self.angle_high = angle_high
        self.rotation_type = rotation_type
      
    def __call__(self, img, kpoints):
        
        '''kpoints - (dimension,N_KPOINTS) array of locations.
        dimension can be 2 or 3'''
        
        angle = np.random.uniform(low=self.angle_low, high=self.angle_high)
        #rotate image 
        img_new = img.rotate(angle)
    
        #rotate keypoints
        angle = -np.deg2rad(angle)
        if self.rotation_type=='3d':
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0,0,1]])
            kpoints_new = np.matmul(rotation_matrix, kpoints)
            
        else:
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])

            w,h = img.size
            kpoints[0,:] = kpoints[0,:] - w/2
            kpoints[1,:] = kpoints[1,:] - h/2
            kpoints_new = np.matmul(rotation_matrix, kpoints)

            kpoints_new[0,:] = kpoints_new[0,:] + w/2
            kpoints_new[1,:] = kpoints_new[1,:] + h/2
        
        return img_new, kpoints_new
    
    
def get_heatmap_from_coordinates(keypoint_arr, shape, is_present, num_keypoints=21):
    keypoint_arr = np.array(np.round(keypoint_arr * shape, 0), dtype='int')
    heatmap = np.zeros((shape, shape, num_keypoints), dtype='uint8')
    
    for i in range(num_keypoints):
        x = keypoint_arr[i,0]
        y = keypoint_arr[i,1]
        if x>=0 and y>=0 and x<shape and y<shape and is_present[i]==1:
            heatmap[y,x,i]=255
        
    return heatmap




# ============= VISUALIZATIONS =============
def visualize_2d(images, vectors_2d):
    
    img_num, _, img_size, _ = images.shape
    plt.figure(figsize=(15,15))
    cols = 4
    rows = np.ceil(img_num/4)
    
    for i in range(img_num):
        img = images[i,:,:,:]
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        
        plt.subplot(cols,rows, i+1)
        plt.imshow(img)
        keypoints = np.reshape(vectors_2d[i,:].numpy() * img_size, (21,2))
        plt.plot(keypoints[:5,0], keypoints[:5,1], 'g') 
        plt.plot(keypoints[1:5,0], keypoints[1:5,1], 'go', ms=3)
        plt.plot(keypoints[[0,5,6,7,8],0], keypoints[[0,5,6,7,8],1], 'c') 
        plt.plot(keypoints[5:9,0], keypoints[5:9,1], 'co', ms=3)
        plt.plot(keypoints[[0,9,10,11,12],0], keypoints[[0,9,10,11,12],1], 'b') 
        plt.plot(keypoints[9:13,0], keypoints[9:13,1], 'bo', ms=3)
        plt.plot(keypoints[[0,13,14,15,16],0], keypoints[[0,13,14,15,16],1], 'm') 
        plt.plot(keypoints[13:17,0], keypoints[13:17,1], 'mo', ms=3)
        plt.plot(keypoints[[0,17,18,19,20],0], keypoints[[0,17,18,19,20],1], 'r') 
        plt.plot(keypoints[17:21,0], keypoints[17:21,1], 'ro', ms=3)
        plt.axis('off')
        #plt.title('Prediction ' + str(i))
        
        plt.imshow(img)
    
    plt.show()    
    return  



def heatmaps_to_coordinates(heatmaps, cuda):
    sums = torch.sum(heatmaps, dim=[2,3])
    sums = sums.unsqueeze(2).unsqueeze(3)

    normalized = heatmaps / sums
    arr = torch.tensor(np.float32(np.arange(0,128))).repeat(16,21,1).cuda(cuda)
    x_prob = torch.sum(normalized, dim=2)
    y_prob = torch.sum(normalized, dim=3)

    x = torch.sum((arr * x_prob), dim=2)
    y = torch.sum((arr * y_prob), dim=2)
    
    vector = torch.cat([x,y], dim=1)
    vector = vector.view(16,2,21).transpose(2,1)
    vector = vector.contiguous().view(16,-1)
    return vector / 128


def visualize_3d(images, vectors_3d):
    
    img_num, _, img_size, _ = images.shape
    for i in range(img_num):
        plt.figure(figsize=(3,3))
        img = images[i,:,:,:]
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        plt.axis('off')
        plt.title('Image '+ str(i))
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        keypoints = np.reshape(vectors_3d[i,:].numpy(), (21,3))
        plt.plot(keypoints[:5,0], keypoints[:5,1], keypoints[:5,2],  'g')
        plt.plot(keypoints[1:5,0], keypoints[1:5,1], keypoints[1:5,2],  'go', ms=3)
        plt.plot(keypoints[[0,5,6,7,8],0], keypoints[[0,5,6,7,8],1], keypoints[[0,5,6,7,8],2],  'c')
        plt.plot(keypoints[5:9,0], keypoints[5:9,1], keypoints[5:9,2],  'co', ms=3)
        plt.plot(keypoints[[0,9,10,11,12],0], keypoints[[0,9,10,11,12],1], keypoints[[0,9,10,11,12],2],  'b')
        plt.plot(keypoints[9:13,0], keypoints[9:13,1], keypoints[9:13,2],  'bo', ms=3)
        plt.plot(keypoints[[0,13,14,15,16],0], keypoints[[0,13,14,15,16],1], keypoints[[0,13,14,15,16],2],  'm')
        plt.plot(keypoints[13:17,0], keypoints[13:17,1], keypoints[13:17,2],  'mo', ms=3)
        plt.plot(keypoints[[0,17,18,19,20],0], keypoints[[0,17,18,19,20],1], keypoints[[0,17,18,19,20],2], 'r')
        plt.plot(keypoints[17:21,0], keypoints[17:21,1], keypoints[17:21,2],  'ro', ms=3)
        plt.title('Prediction '+ str(i))
        plt.show()
        
        
        
        

        
        
        

           