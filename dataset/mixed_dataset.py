import torch
from torch.utils.data import Dataset

from .synth_hands import SynthHandsDataset
from .ganerated import GANeratedDataset


class MixedDataset(Dataset):
    def __init__(self, config):
        
        self.SynthHands = SynthHandsDataset({'path': config['path_synthhands'], 
                                             'path_background': config['path_background'],
                                             'augment': config['augment'],
                                             'scope': config['scope']})
        self.GANerated = GANeratedDataset({'path': config['path_ganerated'], 
                                           'augment': config['augment'],
                                           'scope': config['scope']})
        
        self.len_SynthHands = self.SynthHands.__len__()
        self.len_GANerated = self.GANerated.__len__()
        

    def __getitem__(self, idx):

        if idx < self.len_SynthHands:
            return self.SynthHands.__getitem__(idx)
        else:
            return self.GANerated.__getitem__(idx-self.len_SynthHands)

    
    def __len__(self):
        return self.len_SynthHands + self.len_GANerated 

    
    
