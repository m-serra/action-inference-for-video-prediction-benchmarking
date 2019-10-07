from .base_model import BaseVideoPredictionModel
from .base_model import VideoPredictionModel
from .savp_model import SAVPVideoPredictionModel
from .sv2p_model import SV2PVideoPredictionModel


def get_model_class(model):
    model_mappings = {
        'savp': 'SAVPVideoPredictionModel',
        'savp_vae': 'SAVPVideoPredictionModel',
        'savp_gan': 'SAVPVideoPredictionModel',
        'sv2p': 'SV2PVideoPredictionModel'
    }
    model_class = model_mappings.get(model, model)
    
    model_class = globals().get(model_class)
    
    if model_class is None or not issubclass(model_class, BaseVideoPredictionModel):
        raise ValueError('Invalid model %s' % model)
    return model_class
