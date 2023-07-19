import torch

'''
src_mask = get_pad_mask(src_seq, self.src_pad_idx)
用于产生Encoder的Mask, 它是一列Bool值, 负责把标点mask掉。

trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
用于产生Decoder的Mask, 它是一个矩阵
'''

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask
