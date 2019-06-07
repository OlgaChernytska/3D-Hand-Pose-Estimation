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
        
    
    elif model_name == 'encoder_decoder_tree':
        from .encoder_decoder_tree import EncoderDecoder
        model = EncoderDecoder()
        
    elif model_name == 'jornet':
        from .jornet import JORNet
        model = JORNet()
        
    elif model_name == 'hand_pose25':
        from .hand_pose_25 import HandPose25
        model = HandPose25()
        
    elif model_name == 'modified_resnet':
        from .modified_resnet import ModifiedResNet
        model = ModifiedResNet()
        
    elif model_name == 'my_model1':
        from .my_model1 import MyModel1
        model = MyModel1()
        
    else:
        raise ValueError('Unknown model: {}'.format(model_name))
    
    model = model.cuda(int(config['cuda']))

    return model
