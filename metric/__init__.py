import torch
import torch.nn as nn
import numpy as np

def get_metric(metric_name):
    
    if metric_name == 'roc_auc_2d':
        return roc_auc_2d
    elif metric_name == 'roc_auc_3d':
        return roc_auc_3d
    elif metric_name == 'keypoint_error':
        return keypoint_error

    else:
        raise ValueError('Unknown criterion: {}'.format(criterion_name))
        
        
def roc_auc_2d(sample, output):  #batch average AUC
    batch_size = sample['image'].shape[0]
    n_kpoints = 21

    min_dist = 0
    max_dist = 1
    step = 0.01
    thres_range = np.arange(min_dist, max_dist, step)
    acc_range = np.zeros(len(thres_range))

    for i in range(batch_size):
        actual = np.reshape(sample['vector_2d'][i,:].numpy(), (n_kpoints,2))
        pred = np.reshape(output['vector_2d'][i,:].detach().cpu().numpy(), (n_kpoints,2))
        error = (actual - pred)**2
        error = np.sqrt(error.sum(axis=1))
    
        acc_range_i = []
        for x in thres_range:
            acc = sum(error <= x)
            acc_range_i.append(acc)
        acc_range_i = np.array(acc_range_i)/n_kpoints
        acc_range += acc_range_i
    
    acc_range = acc_range/batch_size
    area = sum(acc_range * step) 
    return {'area': area, 'thres_range': thres_range, 'acc_range': acc_range}


def roc_auc_3d(sample, output):  #strange metric ??
    batch_size = sample['image'].shape[0]
    n_kpoints = 21

    min_dist = 0
    max_dist = 1
    step = 0.01
    thres_range = np.arange(min_dist, max_dist, step)
    acc_range = np.zeros(len(thres_range))

    for i in range(batch_size):
        actual = np.reshape(sample['vector_3d'][i,:].numpy(), (n_kpoints,3))
        pred = np.reshape(output['vector_3d'][i,:].detach().cpu().numpy(), (n_kpoints,3))
        error = (actual - pred)**2
        error = np.sqrt(error.sum(axis=1))
    
        acc_range_i = []
        for x in thres_range:
            acc = sum(error <= x)
            acc_range_i.append(acc)
        acc_range_i = np.array(acc_range_i)/n_kpoints
        acc_range += acc_range_i
    
    acc_range = acc_range/batch_size
    area = sum(acc_range * step)
    return {'area': area, 'thres_range': thres_range, 'acc_range': acc_range}


def keypoint_error(sample, output, label='vector_2d', is_real=False):
    batch_size = sample['image'].shape[0]
    n_kpoints = 21
    
    if label=='vector_2d':
        dim = 2
    else:
        dim=3
        
    error_keypoint_acc = np.zeros(n_kpoints)
    present_counter = np.zeros(n_kpoints)
    
    if is_real:
        img_size_arr = sample['img_size'].data.tolist()
        
    for i in range(batch_size):
        is_present = sample['is_present'][i].numpy()
        actual = np.reshape(sample[label][i,:].numpy(), (n_kpoints,dim))
        pred = np.reshape(output[label][i,:].detach().cpu().numpy(), (n_kpoints,dim))
        error = (actual - pred)**2
        error = np.sqrt(error.sum(axis=1))
        error = error * is_present
        
        #if is_real:
        #    error = error * img_size_arr[i]
        
        error_keypoint_acc += error
        present_counter += is_present
    
    
    if is_real:
        return {'keypoint_error': error_keypoint_acc, 
                'present_counter': present_counter}
        
        
    else: 
        error_keypoint_acc = error_keypoint_acc / batch_size
        error_total_acc = np.mean(error_keypoint_acc)
    
        return {'keypoint_error': error_keypoint_acc, 
                'total_error': error_total_acc}
        
        