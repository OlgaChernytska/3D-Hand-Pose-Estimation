import argparse
import yaml

import warnings
import torch
import numpy as np

from model import get_model
from dataset import get_dataloader
from optimizer import get_optimizer
from criterion import get_criterion


class Trainer:
    def __init__(self, config):
 
        self.cuda = int(config['cuda'])
        #torch.cuda.empty_cache()
        self.train_dataloader = get_dataloader(config, scope='train')
        self.val_dataloader = get_dataloader(config, scope='val')
        
        self.model = get_model(config)
        try:
            model_weights = 'experiment/' + config['dir'] + '/' + config['weights']
            self.model.load_state_dict(torch.load(model_weights)['model'])
            print('Weigths loaded')
        except:
            print('Weights randomized')

        self.optimizer = get_optimizer(config, self.model)
        self.total_epochs = config['epochs']
        self.batches_per_epoch = config['batches_per_epoch']
        self.val_batches_per_epoch = config['val_batches_per_epoch']
        
        self.final_weights_file = 'experiment/' + config['dir'] + '/weights_last.pth' 
        self.best_weights_file = 'experiment/' + config['dir'] + '/weights_best.pth'
        self.log_file = 'experiment/' + config['dir'] + '/logs.csv' 
        
        self.loss_dict = {'sample_name': config['sample_name'],
                          'output_name': config['output_name'],
                          'loss': [get_criterion(x) for x in config['loss_criterion']],
                          'weight': config['loss_weight']}
        
        
        
        self.train_fe = bool(config['train_feature_extractor'])
        
    def train(self):
        
        if not self.train_fe:
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.layer1.parameters():
                param.requires_grad = False
            for param in self.model.layer2.parameters():
                param.requires_grad = False
            for param in self.model.layer3.parameters():
                param.requires_grad = False
            for param in self.model.layer3.parameters():
                param.requires_grad = False
   
        best_val_loss = 10000
    
        for epoch in range(self.total_epochs):
            
            batches = 0
            logging = {}

            for sample in self.train_dataloader:
                self.model.train()
                self.optimizer.zero_grad()
                
                sample['image'] = sample['image'].cuda(self.cuda)
                output = self.model(sample)
                loss_data = self._loss(sample, output)  
                
                loss = loss_data['total']
                loss.backward()
                self.optimizer.step()
                
                # logging
                for key, value in loss_data.items():
                    try: 
                        logging[key].append(value.item())
                    except:
                        logging[key] = [value.item()]
              
                batches += 1
                if batches >= self.batches_per_epoch:
                    break
                    
                  
            val_logging = self._validate()
            mean_train_loss = np.mean(logging['total'])
            mean_val_loss = np.mean(val_logging['total'])
            print('====Epoch {}. Train loss: {}. Val loss: {}'.format(epoch, mean_train_loss, mean_val_loss))
            
            
            # logging
            if epoch>0: 
                with open(self.log_file, 'a') as fp:
                    fp.write(str(epoch+1))
                    for key, value in logging.items():
                        fp.write(',' + str(np.mean(value)))
                    for key, value in val_logging.items():
                        fp.write(',' + str(np.mean(value)))
                    fp.write('\n') 
                    
            else:
                with open(self.log_file, 'w') as fp:
                    fp.write('epoch')
                    for key, value in logging.items():
                        fp.write(',train_' + key)   
                    for key, value in logging.items():   
                        fp.write(',val_' + key)
                    fp.write('\n')
                    
                    fp.write(str(epoch+1))
                    for key, value in logging.items():
                        fp.write(',' + str(np.mean(value)))
                    for key, value in val_logging.items():
                        fp.write(',' + str(np.mean(value)))
                    fp.write('\n')    
                    

            # saving model
            torch.save({
                'model': self.model.state_dict()
            }, self.final_weights_file)
            
            #if mean_val_loss < best_val_loss:
            #    best_val_loss = mean_val_loss
            #    torch.save({
            #        'model': self.model.state_dict()
            #    }, self.best_weights_file)
            
            
            if (epoch+1) % 20 == 0:
                torch.save({
                    'model': self.model.state_dict()
                }, 'experiment/' + config['dir'] + '/weights_' + str(epoch+1).zfill(3) + '.pth' )
            

    
    def _validate(self):
        self.model.eval()
        
        batches = 0
        logging = {}
        with torch.no_grad():
            for sample in self.val_dataloader:
                
                sample['image'] = sample['image'].cuda(self.cuda)
                output = self.model(sample)
                loss_data = self._loss(sample, output)  ##add logging intermediate losses
                loss = loss_data['total']
                
                for key, value in loss_data.items():
                    try: 
                        logging[key].append(value.item())
                    except:
                        logging[key] = [value.item()]
                
                batches += 1
                if batches >= self.val_batches_per_epoch:
                    break
                    
        return logging
    
    
    def _loss(self, sample, output):
        loss = 0
        return_dict = {}
        
        for i, l_name in enumerate(self.loss_dict['output_name']):
            sample_name = self.loss_dict['sample_name'][i]
            
            if l_name == 'heatmaps':
                img_size = sample['heatmaps'].shape[2]
                pres = sample['is_present'].unsqueeze(2).unsqueeze(2)
                pres = pres.repeat([1,1,img_size,img_size]).cuda(self.cuda)
                output = torch.mul(output[l_name],pres) 
                inter_loss = self.loss_dict['loss'][i] (output, 
                                                sample[sample_name].cuda(self.cuda))

            else:
                inter_loss = self.loss_dict['loss'][i] (output[l_name], 
                                                sample[sample_name].cuda(self.cuda)) 
            
            return_dict[l_name] = inter_loss
            loss += inter_loss * self.loss_dict['weight'][i]       
        
        return_dict['total'] = loss
        return return_dict


if __name__ == '__main__':
    
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('exper_folder', help='Provide experiment folder')
    args = parser.parse_args()
    
    print('Experiment {} started'.format(args.exper_folder))
    
    config_file = 'experiment/' + args.exper_folder + '/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
   
    config['dir'] = args.exper_folder
    trainer = Trainer(config)
    trainer.train()
    
    print('Experiment {} ended'.format(args.exper_folder))