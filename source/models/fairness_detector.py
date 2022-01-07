import torch
import torch.nn as nn
import numpy as np
import torchvision 
from torchvision import datasets, models, transforms
from source.logging import log
from source.models.interface.model_interface import ModelInterface

class FairnessDetector(ModelInterface):
    def __init__(self,
                model_path='/assets/res34_fair_align_multi_4_20190809.pt',
                model_loader=torch.load,
                device=torch.device('cpu')):
        self.model_path = model_path
        self._device = device
        log(f'Building pretrained resnet34 architecture', 'D')
        self.model = torchvision.models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)
        
        log(f'Loading resnet model from {model_path}', 'D')
        self.model.load_state_dict(model_loader(model_path, map_location=device))
        
        log(f'Sending model to {device.type}', 'D')
        self.model = self.model.to(device)
        self.model.eval()
        
    
        log(f'Setting image preprocess method', 'D')
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
