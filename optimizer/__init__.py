import torch


def get_optimizer(config, model):
        lr = config['lr']
        optim_name = config['optimizer']
        if optim_name == 'sgd':
            optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError('Unknown optimizer: {}'.format(optim_name))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
        
        return optimizer