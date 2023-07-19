from attentions import *

class EncodeLayer(nn.Module):
    def __init__(self, d_model, dim_pk, dim_v, n_head, dim_hid, dropout=0.1):
        super(EncodeLayer, self).__init__()
        self.multiattn = MultiHeadAttention(d_model, dim_pk, dim_v, n_head, dropout)
        self.feedforward = FeedForward(d_model, dim_hid, dropout)
    
    def forward(self, q, k, v, mask=None):
        attn_out, attn = self.multiattn(q, k, v, mask)
        encode_out = self.feedforward(attn_out)
        return encode_out, attn


class DecodeLayer(nn.Module):
    def __init__(self, d_model, dim_pk, dim_v, n_head, dim_hid, dropout=0.1):
        super(DecodeLayer, self).__init__()
        self.selfattn = MultiHeadAttention(d_model, dim_pk, dim_v, n_head, dropout)
        self.crossattn = MultiHeadAttention(d_model, dim_pk, dim_v, n_head, dropout)
        self.feedforward = FeedForward(d_model, dim_hid, dropout)
    
    def forward(self, enc_out, dec_in, self_mask=None, cross_mask=None):
        selfattn_out, self_attn = self.selfattn(enc_out, enc_out, enc_out, self_mask)
        crossattn_out, cross_attn = self.crossattn(dec_in, selfattn_out, selfattn_out, cross_mask)
        decode_out = self.feedforward(crossattn_out)
        return decode_out, self_attn, cross_attn



if __name__ == '__main__':
    q = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3]]]).float()
    k = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]]]).float()
    v = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]]]).float()
    
    enc = EncodeLayer(q.size(-1), 4, 4, 8, q.size(-1)*2)
    enc_out, _ = enc(q, k, v)
    print(enc_out, enc_out.size())
    
    dec = DecodeLayer(q.size(-1), 4, 4, 8, q.size(-1)*2)
    dec_out, _, _ = dec(enc_out, q)
    print(dec_out, dec_out.size())