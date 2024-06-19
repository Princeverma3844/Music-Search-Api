import torch
from msclap import CLAP

def get_clap_model():
    clap_model = CLAP(version='2023', use_cuda=False)
    clap_model.clap.load_state_dict(torch.load("./clap_model_weight.pth/clap_model_weight.pth", map_location=torch.device('cpu')))
    return clap_model
