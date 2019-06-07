import yaml
import torch
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

from model import get_model
from dataset import get_dataloader
from metric import get_metric

def evaluate(config):
    model = get_model(config)
    weight_file = 'experiment/' + config['dir'] + '/' + config['weights']
    model.load_state_dict(torch.load(weight_file)['model'])
    model.eval()
    n_kpoints = 21
    
    is_real = bool(config['is_real'])
    
    cuda = config['cuda']
    path_dir = 'experiment/' + config['dir'] + '/' 
    path_val_file = path_dir + config['dataset'] + '_valuation.json'

    dataloader = get_dataloader(config, scope='val')
    roc_auc_2d = get_metric('roc_auc_2d')
    roc_auc_3d = get_metric('roc_auc_3d')
    keypoint_error = get_metric('keypoint_error')

    area_2d = 0
    error_keypoint_2d = np.zeros(n_kpoints)
    error_keypoint_total_2d = 0
    
    area_3d = 0
    error_keypoint_3d = np.zeros(n_kpoints)
    error_keypoint_total_3d = 0
    
    present_counter = np.zeros(n_kpoints)
    
    thres_range = np.arange(0, 1, 0.01)
    acc_range_2d = np.zeros(len(thres_range))
    acc_range_3d = np.zeros(len(thres_range))

    for num, sample in enumerate(dataloader, 1):
        
        sample['image'] = sample['image'].cuda(cuda)
        output = model(sample)
        
        if is_real:
            batch_avg_keypoint_error_2d = keypoint_error(sample, output, 'vector_2d', True)
            error_keypoint_2d += batch_avg_keypoint_error_2d['keypoint_error']
            present_counter += batch_avg_keypoint_error_2d['present_counter']
        
        
        
        else:
            
            batch_avg_metric_2d = roc_auc_2d(sample, output)
            area_2d += batch_avg_metric_2d['area']
            acc_range_2d += batch_avg_metric_2d['acc_range']
            
            batch_avg_keypoint_error_2d = keypoint_error(sample, output, 'vector_2d', False)
            error_keypoint_2d += batch_avg_keypoint_error_2d['keypoint_error']
            error_keypoint_total_2d += batch_avg_keypoint_error_2d['total_error']
        
            batch_avg_metric_3d = roc_auc_3d(sample, output)
            area_3d += batch_avg_metric_3d['area']
            acc_range_3d += batch_avg_metric_3d['acc_range']
        
            batch_avg_keypoint_error_3d = keypoint_error(sample, output, 'vector_3d', False)
            error_keypoint_3d += batch_avg_keypoint_error_3d['keypoint_error']
            error_keypoint_total_3d += batch_avg_keypoint_error_3d['total_error']
        
        if num % 50 == 0:
            print('Evaluation done for {} batches'.format(num))
        
 
    
    if is_real:
        
        error_keypoint_2d_avg = np.array([-1.]* 21)
        for i in range(21):
            if not present_counter[i] == 0:
                error_keypoint_2d_avg[i] = error_keypoint_2d[i] / present_counter[i]
                
        error_total_2d_avg = np.sum(error_keypoint_2d) / np.sum(present_counter)
    
        
        val_dict = {
                'error_keypoint_2d': list(error_keypoint_2d_avg),
                'error_total_2d': error_total_2d_avg
                }
        
    else:
        val_dict = {'thres_range': list(thres_range),
                'area_2d': area_2d / num,
                'acc_range_2d': list(acc_range_2d / num),
    
                'area_3d': area_3d / num,
                'acc_range_3d': list(acc_range_3d / num),
    
                'error_keypoint_2d': list(error_keypoint_2d / num),
                'error_total_2d': error_keypoint_total_2d / num, 
    
                'error_keypoint_3d': list(error_keypoint_3d / num),
                'error_total_3d': error_keypoint_total_3d / num
                }
    
    
    
    # saving data
    with open(path_val_file, 'w') as fp:
        json.dump(val_dict, fp)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exper_folder', help='Provide experiment folder')
    parser.add_argument('dataset', help='Provide dataset')
    args = parser.parse_args()
    
    print('Evaluation {} started'.format(args.exper_folder))
    
    config_file = 'experiment/' + args.exper_folder + '/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    config['dir'] = args.exper_folder
    config['dataset'] = args.dataset
    
    
    if args.dataset in ['stereo', 'dexter+object']:
        config['is_real'] = 1
    else:
        config['is_real'] = 0
    
    with torch.no_grad():
        evaluate(config)
        
    print('Evaluation {} ended'.format(args.exper_folder))

