
from modeling_t5 import UMIE
from umie_model import UMIEModel
import torch.nn as nn



class UMIEMultiTask(UMIE):
    def __init__(self, config):
        super().__init__(config)


        

    def train_step(self, batch, **kwargs):
        return UMIEModel.train_step(self, batch, **kwargs)

            
        
    def valid_step(self, batch, **kwargs):
        return UMIEModel.test_step(self, batch, **kwargs)

        
        
            
    def test_step(self, batch, **kwargs):
        return UMIEModel.test_step(self, batch, **kwargs)

            
        
        