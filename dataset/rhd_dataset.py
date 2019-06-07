import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image
    
class RhdDatasetSegmentation(Dataset):
    '''Link to dataset: https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html'''
    
    def __init__(self, config):
        
        self.scope = config['scope']
        
        if self.scope == 'train':
            self.root_dir = config['path'] + 'training/'
        else:
            self.root_dir = config['path'] + 'evaluation/'
            
        self.transform_image = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
        if self.scope == 'train':
            self.transform_augment = transforms.Compose([
                    RandomCrop(256),  
                    RandomHolizontalFlip(0.3)])
        
        else:
            self.transform_augment = transforms.Compose([
                    Resize(256)])
            
        self.transform_mask = transforms.ToTensor()
        self.samples = np.sort(os.listdir(self.root_dir + 'color'))
        
    def __getitem__(self, idx):
        
        image =  Image.open(self.root_dir + 'color/' + self.samples[idx])
        mask = Image.open(self.root_dir + 'mask/' + self.samples[idx])
        mask = np.array(mask)
        mask[mask==1]=0
        mask[mask>1]=255
        mask = Image.fromarray(mask)
        
        sample = {'image': image,
                  'target': mask}
        
        sample = self.transform_augment(sample)
        image = self.transform_image(sample['image'])
        mask = self.transform_mask(sample['target'])
            
        return {'image': image, 'target': mask}

    def __len__(self):
        return len(self.samples)
    


    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = np.array(sample['image']), np.array(sample['target'])

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                      left: left + new_w]

        return {'image': Image.fromarray(image), 'target': Image.fromarray(mask)}
    
class RandomHolizontalFlip(object):
    """Flip randomly the image in a sample.

    Args:
        prob: Probability of each image to be flipped.
    """

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand()<self.prob:
            image, mask = np.array(sample['image']), np.array(sample['target'])
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            return {'image': Image.fromarray(image), 'target': Image.fromarray(mask)}
        else:
            return sample
        
class Resize(object):
    """Resizes the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['target']
        image = image.resize(self.output_size)
        mask = mask.resize(self.output_size)
        
        return {'image': image, 'target': mask}


