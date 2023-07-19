import torch
import torch.nn as nn
import numpy as np


class PositionEncoding(nn.Module):
    def __init__(self, d_model, n_position):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.n_position = n_position
        self.register_buffer('pos_table', self.get_sinusoid_encoding())
    
    def get_sinusoid_encoding(self):
        def get_encoding_vec(pos):
            return np.array([pos / np.power(1e4, 2*(i//2)/self.d_model) for i in range(self.d_model)])
        
        sinusoid_table = np.array([get_encoding_vec(pos) for pos in range(self.n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        
        return torch.tensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, x):
        '''
        x: [B, N, D]
        '''
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    
    

if __name__ == '__main__':
    x = torch.tensor([[1,2,3], [4,5,6]])
    PE = PositionEncoding(3, 2)
    print(PE(x))