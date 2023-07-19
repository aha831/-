import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        return output, attn
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, dim_qk, dim_v, n_head, dropout=0.1):
        super(MultiHeadAttention, self). __init__()
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.n_head = n_head
        self.d_model = d_model

        self.wqs = nn.Linear(self.d_model, self.dim_qk*self.n_head, bias=False)
        self.wks = nn.Linear(self.d_model, self.dim_qk*self.n_head, bias=False)
        self.wvs = nn.Linear(self.d_model, self.dim_v*self.n_head, bias=False)
        self.fc = nn.Linear(self.dim_v*self.n_head, self.d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(self.d_model, eps=1e-6)
        
        self.dot_attn = ScaledDotProductAttention(self.d_model)
    
    def forward(self, q, k, v, mask=None):
        '''
            q/k/v: [B, N_q/N_k/N_v, d_model]
        '''
        res = q
        B, N_q, N_k, N_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.wqs(q).view(B, N_q, self.n_head, -1).transpose(1, 2)
        k = self.wks(k).view(B, N_k, self.n_head, -1).transpose(1, 2)
        v = self.wvs(v).view(B, N_v, self.n_head, -1).transpose(1, 2)
        
        q, attn = self.dot_attn(q, k, v, mask)
        q = self.dropout(self.fc(q.transpose(1, 2).contiguous().view(B, N_q, -1)))
        
        res += q
        res = self.layernorm(res)
        
        return res, attn

class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_hid, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(dim_in, dim_hid, bias=False)
        self.fc2 = nn.Linear(dim_hid, dim_in, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_in, eps=1e-6)
        
    
    def forward(self, x):
        output = x
        x = self.dropout(self.fc2(F.relu(self.fc1(x))))
        
        output += x
        output = self.norm(output)

        return output




if __name__ == '__main__':
    q = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3]]]).float()
    k = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]]]).float()
    v = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]]]).float()
    
    mul_attn = MultiHeadAttention(q.size(-1), 4, 4, 8)
    out = mul_attn(q, k, v)[0]
    print(out, out.size())
    
    ff = FeedForward(q.size(-1), 12, dropout=0.1)
    out = ff(out)
    print(out, out.size())