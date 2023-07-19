from attentions import *
from pe import *
from utils import *

class EncodeLayer(nn.Module):
    def __init__(self, d_model, dim_pk, dim_v, n_head, dim_hid, dropout=0.1):
        super(EncodeLayer, self).__init__()
        self.multiattn = MultiHeadAttention(d_model, dim_pk, dim_v, n_head, dropout)
        self.feedforward = FeedForward(d_model, dim_hid, dropout)
    
    def forward(self, enc_input, mask=None):
        attn_out, attn = self.multiattn(enc_input, enc_input, enc_input, mask)
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



class Encoder(nn.Module):
    def __init__(self, dic_size, n_layer, d_model, d_hid, d_qk, d_v, n_head, n_position, pad_idx, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(dic_size, d_model, padding_idx=pad_idx)
        self.pe = PositionEncoding(d_model, n_position)
        # self.encodelayer = EncodeLayer(d_model, d_qk, d_v, n_head, d_hid, dropout)
        self.encoder = nn.ModuleList([EncodeLayer(d_model, d_qk, d_v, n_head, d_hid, dropout) for _ in range(n_layer)])
        
        self.droput = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        
    def forward(self, sec_input, enc_mask=None):
        '''
        sec_input: [B, N]
        '''
        enc_out = self.droput(self.pe(self.embedding(sec_input)))
        enc_attns = []
        for layer in self.encoder:
            enc_out, tmp_attn = layer(enc_out, enc_mask)
            enc_attns.append(tmp_attn)
        
        return enc_out, enc_attns



class Decoder(nn.Module):
    def __init__(self, dic_size, n_layer, d_model, d_hid, d_qk, d_v, n_head, n_position, pad_idx, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(dic_size, d_model, padding_idx=pad_idx)
        self.pe = PositionEncoding(d_model, n_position)
        self.decoder = nn.ModuleList([DecodeLayer(d_model, d_qk, d_v, n_head, d_hid, dropout) for _ in range(n_layer)])
        
        self.droput = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        
    def forward(self, enc_out, dec_input, dec_self_mask=None, dec_cross_mask=None):
        '''
        enc_out: [B, N, D]
        dec_input: [B, N]
        '''
        dec_out = self.droput(self.pe(self.embedding(dec_input)))
        dec_self_attns, dec_cross_attns = [], []
        for layer in self.decoder:
            dec_out, tmp_self_attn, tmp_cross_attn = layer(enc_out, dec_out, dec_self_mask, dec_cross_mask)
            dec_self_attns.append(tmp_self_attn)
            dec_cross_attns.append(tmp_cross_attn)
        
        return dec_out, dec_self_attns, dec_cross_attns
        
        
class Transformer(nn.Module):
    ''' 
    整体的Transformer, 直接搬运的代码, 主要就是最后加了一个对decoder输出的fc计算, 以及model参数的init
    A sequence to sequence model with attention mechanism. 
    '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
    


if __name__ == '__main__':
    # q = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3]]]).float()
    # k = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]]]).float()
    # v = torch.tensor([[[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]], [[1,2,3,1,2,3], [4,5,6,1,2,3], [7,8,9,1,2,3]]]).float()
    
    # enc = EncodeLayer(q.size(-1), 4, 4, 8, q.size(-1)*2)
    # enc_out, _ = enc(q, k, v)
    # print(enc_out, enc_out.size())
    
    # dec = DecodeLayer(q.size(-1), 4, 4, 8, q.size(-1)*2)
    # dec_out, _, _ = dec(enc_out, q)
    # print(dec_out, dec_out.size())
    
    sec_inp = torch.tensor([[0,1,4,2,5,5,5,5], [0,1,4,2,2,3,5,5]])
    enc = Encoder(6, 4, 6, 12, 4, 6, 8, 8, pad_idx=5).float()
    enc_out, _ = enc(sec_inp)
    print(enc_out, enc_out.size())
    
    dec_inp = torch.tensor([[1,0,2,4,5,5,5,5], [1,0,2,4,4,0,5,5]])
    dec = Decoder(6, 4, 6, 12, 4, 6, 8, 8, pad_idx=5).float()
    dec_out, _, _ = dec(enc_out, dec_inp)
    print(dec_out, dec_out.size())