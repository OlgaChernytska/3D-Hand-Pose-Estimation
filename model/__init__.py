import torch


def get_model(config):
    model_name = config['model']
    
    if model_name == 'model_direct_vector':
        from .model1_direct_vector import EncoderDecoder
        model = EncoderDecoder()
        
    elif model_name == 'model_latent_vector':
        from .model2_latent_vector import EncoderDecoder
        model = EncoderDecoder()
        
    elif model_name == 'model_direct_tree':
        from .model3_direct_tree import EncoderDecoder
        model = EncoderDecoder()   

        
    else:
        raise ValueError('Unknown model: {}'.format(model_name))
    
    model = model.cuda(int(config['cuda']))

    return model
