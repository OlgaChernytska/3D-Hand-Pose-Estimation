import torch
from torch.utils.data import Dataset

from .stereo_dataset import StereoDataset
from .dexter_object import DexterObjectDataset


class MixedDataset(Dataset):
    def __init__(self, config):
        
        self.Stereo = StereoDataset({'path': config['path_stereo'], 
                                     'augment': config['augment'],
                                     'scope': config['scope']})
        
        self.DexterObject = DexterObjectDataset({'path': config['path_dexter'], 
                                                 'augment': config['augment'],
                                                 'scope': config['scope']})
        
        self.len_Stereo = self.Stereo.__len__()
        self.len_DexterObject = self.DexterObject.__len__()
        

    def __getitem__(self, idx):

        if idx < self.len_Stereo:
            return self.Stereo.__getitem__(idx)
        else:
            return self.DexterObject.__getitem__(idx-self.len_Stereo)

    
    def __len__(self):
        return self.len_Stereo + self.len_DexterObject

    
    
