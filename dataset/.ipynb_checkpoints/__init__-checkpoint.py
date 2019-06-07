from torch.utils.data import DataLoader
import yaml

def get_dataloader(config, scope):
    dataset = config['dataset']
    augmentation = config['augment']
    
    with open('__local__config.yaml', 'r') as f:
        local_config = yaml.load(f)
    path_to_data = local_config['data_path']
    

    if dataset == 'synth_hands':
        from .synth_hands import SynthHandsDataset
        dataset = SynthHandsDataset({'path': path_to_data + 'SynthHands_Release/', 
                                     'path_background': path_to_data + 'ADE20K_2016_07_26/images/',
                                     'augment': augmentation,
                                     'scope': scope})
    
    elif dataset == 'ganerated':
        from .ganerated import GANeratedDataset
        dataset = GANeratedDataset({'path': path_to_data + 'GANeratedHands_Release/data/', 
                                    'augment': augmentation,
                                    'scope': scope})
    
    elif dataset == 'mixed_dataset':
        from .mixed_dataset import MixedDataset
        dataset = MixedDataset({'path_synthhands': path_to_data + 'SynthHands_Release/',
                                'path_ganerated': path_to_data + 'GANeratedHands_Release/data/',
                                'path_background': path_to_data + 'ADE20K_2016_07_26/images/',
                                'augment': augmentation,
                                'scope': scope})
        
    elif dataset == 'mixed_dataset2':
        from .mixed_dataset2 import MixedDataset
        dataset = MixedDataset({'path_synthhands': path_to_data + 'SynthHands_Release/',
                                'path_ganerated': path_to_data + 'GANeratedHands_Release/data/',
                                'path_background': path_to_data + 'ADE20K_2016_07_26/images/',
                                'augment': augmentation,
                                'scope': scope})
        
    elif dataset == 'stereo':
        from .stereo_dataset import StereoDataset
        dataset = StereoDataset({'path': path_to_data + 'stereohandtracking/', 
                                 'augment': augmentation,
                                 'scope': scope})
        
    elif dataset == 'dexter+object':
        from .dexter_object import DexterObjectDataset
        dataset = DexterObjectDataset({'path': path_to_data + 'dexter+object/', 
                                       'augment': augmentation,
                                       'scope': scope})
        
    elif dataset == 'mixed_dataset_real':
        from .mixed_dataset_real import MixedDataset
        dataset = MixedDataset({'path_stereo': path_to_data + 'stereohandtracking/',
                                'path_dexter': path_to_data + 'dexter+object/',
                                'augment': augmentation,
                                'scope': scope})
        
   
    
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
        
    
    return DataLoader(dataset,
                      config['batch_size'],
                      shuffle=bool(config['shuffle']), 
                      num_workers=8,
                      drop_last=True)