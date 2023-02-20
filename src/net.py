# Several functions borrowed from https://github.com/ermongroup/CSDI/blob/main/diff_models.py
# And also from Martinez et al: https://github.com/una-dinosauria/human-motion-prediction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# make time t into 128-dim embedding vector.
# borrowed from https://github.com/ermongroup/CSDI/blob/main/diff_models.py 
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        # Codes from CSDI
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


# Positional Encoding modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TFED_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.enc_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.enc_temporal_TF = nn.TransformerEncoder(self.enc_layer, num_layers=num_layers)
        self.enc_temporal_in = nn.Linear(input_dim, qkv_dim)
        self.enc_temporal_out = nn.Linear(qkv_dim, qkv_dim)

        self.dec_layer = nn.TransformerDecoderLayer(d_model=qkv_dim, nhead=num_heads, 
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.dec_TF = nn.TransformerDecoder(self.dec_layer, num_layers=num_layers)
        self.dec_inp = nn.Linear(input_dim, qkv_dim)
        self.dec_feat = nn.Linear(qkv_dim, qkv_dim)
        self.dec_out = nn.Linear(qkv_dim, input_dim)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
        self.step_encoder = DiffusionEmbedding(diff_steps, qkv_dim)

    def encode(self, x):
        L, B, D = x.shape

        temporal_inp = x 
        temporal_inp = self.enc_temporal_in(temporal_inp)
        temporal_out = self.pos_encoder(temporal_inp)
        temporal_out = self.enc_temporal_TF(temporal_out)
        temporal_out = self.enc_temporal_out(temporal_out)
        
        return temporal_out 

    def decode(self, noise, feat, t):
        step_embed = self.step_encoder(t) # B D

        feat = self.dec_feat(feat)
        tgt = self.pos_encoder(self.dec_inp(noise) + step_embed.unsqueeze(0))
        out = self.dec_TF(tgt=tgt, memory=feat)
        res = self.dec_out(out)
        return res

    def forward(self, noise, x, t):
        return self.decode(noise, self.encode(x), t)


class TF2CHED_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.combine_layer = nn.Conv2d(2, 1, 1, 1, 0)

        ## decoder
        self.dec_layer = nn.TransformerDecoderLayer(d_model=qkv_dim, nhead=num_heads, 
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.dec_TF = nn.TransformerDecoder(self.dec_layer, num_layers=num_layers)
        self.dec_inp = nn.Linear(input_dim, qkv_dim)
        self.dec_feat = nn.Linear(input_dim, qkv_dim)
        self.dec_out = nn.Linear(qkv_dim, input_dim)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
        self.step_encoder_tgt = DiffusionEmbedding(diff_steps, qkv_dim)
        self.step_encoder_feat = DiffusionEmbedding(diff_steps, qkv_dim)

    def encode(self, x):
        temporal = self.temporal_inp_fc(x)
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)

        spatial = self.spatial_inp_fc(x.permute(2, 1, 0)) # L B D -> D B L 
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial)

        feat = torch.cat([self.temporal_out_fc(temporal).unsqueeze(1),
                          self.spatial_out_fc(spatial).permute(2, 1, 0).unsqueeze(1)], dim=1)
        out = self.combine_layer(feat).squeeze(1)
        return out
    
    def decode(self, noise, feat, t):
        step_embed_feat = self.step_encoder_feat(t) # B D
        step_embed_tgt = self.step_encoder_tgt(t) # B D

        feat = self.dec_feat(feat) + step_embed_feat.unsqueeze(0)
        tgt = self.pos_encoder(self.dec_inp(noise) + step_embed_tgt.unsqueeze(0))
        out = self.dec_TF(tgt=tgt, memory=feat)
        res = self.dec_out(out)
        return res

    def forward(self, noise, x, t):
        return self.decode(noise, self.encode(x), t)


class STTF_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len+pred_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len+pred_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
        self.step_temporal_encoder = DiffusionEmbedding(diff_steps, qkv_dim)
        self.step_spatial_encoder = DiffusionEmbedding(diff_steps, qkv_dim)

    def forward(self, noise, x, t):
        step_spatial_embed = self.step_spatial_encoder(t)
        step_temporal_embed = self.step_temporal_encoder(t)
        window = torch.cat([x, noise], dim=0)

        spatial = self.spatial_inp_fc(window.permute(2, 1, 0)) + step_spatial_embed # L B D -> D B L 
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial) 
        spatial = self.spatial_out_fc(spatial).permute(2, 1, 0)

        temporal = self.temporal_inp_fc(spatial) + step_temporal_embed
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)
        temporal = self.temporal_out_fc(temporal)

        return temporal[x.shape[0]:]

class STTF_Supervised(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len+pred_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len+pred_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
    def forward(self, window):
        spatial = self.spatial_inp_fc(window.permute(2, 1, 0))
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial) 
        spatial = self.spatial_out_fc(spatial).permute(2, 1, 0)

        temporal = self.temporal_inp_fc(spatial)
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)
        temporal = self.temporal_out_fc(temporal)

        return temporal[self.prefix_len:]

class TSTF_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len+pred_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len+pred_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
        self.step_temporal_encoder = DiffusionEmbedding(diff_steps, qkv_dim)
        self.step_spatial_encoder = DiffusionEmbedding(diff_steps, qkv_dim)

    def forward(self, noise, x, t):
        step_spatial_embed = self.step_spatial_encoder(t)
        step_temporal_embed = self.step_temporal_encoder(t)
        window = torch.cat([x, noise], dim=0)

        temporal = self.temporal_inp_fc(window) + step_temporal_embed
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)
        temporal = self.temporal_out_fc(temporal)

        spatial = self.spatial_inp_fc(temporal.permute(2, 1, 0)) + step_spatial_embed # L B D -> D B L 
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial) 
        spatial = self.spatial_out_fc(spatial).permute(2, 1, 0)

        return spatial[x.shape[0]:]



class TF2CH_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len+pred_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len+pred_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.combine_layer = nn.Conv2d(2, 1, 1, 1, 0)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
        self.step_temporal_encoder = DiffusionEmbedding(diff_steps, qkv_dim)
        self.step_spatial_encoder = DiffusionEmbedding(diff_steps, qkv_dim)

    def forward(self, noise, x, t):
        step_spatial_embed = self.step_spatial_encoder(t)
        step_temporal_embed = self.step_temporal_encoder(t)
        window = torch.cat([x, noise], dim=0)

        temporal = self.temporal_inp_fc(window) + step_spatial_embed
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)

        spatial = self.spatial_inp_fc(window.permute(2, 1, 0)) + step_temporal_embed # L B D -> D B L 
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial)

        feat = torch.cat([self.temporal_out_fc(temporal).unsqueeze(1),
                          self.spatial_out_fc(spatial).permute(2, 1, 0).unsqueeze(1)], dim=1)
        out = self.combine_layer(feat).squeeze(1)
        return out[x.shape[0]:]


class TF2CH_Supervised(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len

        self.spatial_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.spatial_TF = nn.TransformerEncoder(self.spatial_layer, num_layers=num_layers)
        self.spatial_inp_fc = nn.Linear(prefix_len+pred_len, qkv_dim)
        self.spatial_out_fc = nn.Linear(qkv_dim, prefix_len+pred_len)

        self.temporal_layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.temporal_TF = nn.TransformerEncoder(self.temporal_layer, num_layers=num_layers)
        self.temporal_inp_fc = nn.Linear(input_dim, qkv_dim)
        self.temporal_out_fc = nn.Linear(qkv_dim, input_dim)

        self.combine_layer = nn.Conv2d(2, 1, 1, 1, 0)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
    def forward(self, window):
        temporal = self.temporal_inp_fc(window) 
        temporal = self.pos_encoder(temporal)
        temporal = self.temporal_TF(temporal)

        spatial = self.spatial_inp_fc(window.permute(2, 1, 0))
        spatial = self.pos_encoder(spatial)
        spatial = self.spatial_TF(spatial)

        feat = torch.cat([self.temporal_out_fc(temporal).unsqueeze(1),
                          self.spatial_out_fc(spatial).permute(2, 1, 0).unsqueeze(1)], dim=1)
        out = self.combine_layer(feat).squeeze(1)
        
        return out[self.prefix_len:] + window[self.prefix_len:]


class TF_Denoiser(nn.Module):
    def __init__(self, input_dim, qkv_dim, num_layers, num_heads, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.qkv_dim = qkv_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prefix_len = prefix_len
        self.pred_len = pred_len
        self.diff_steps = diff_steps

        self.layer = nn.TransformerEncoderLayer(d_model=qkv_dim, nhead=num_heads,
                                                    dim_feedforward=qkv_dim, activation='gelu')
        self.TF = nn.TransformerEncoder(self.layer, num_layers=num_layers)
        self.inp_fc = nn.Linear(input_dim, qkv_dim)
        self.out_fc = nn.Linear(qkv_dim, input_dim)

        self.pos_encoder = PositionalEncoding(qkv_dim)
        
        self.step_encoder = DiffusionEmbedding(diff_steps, qkv_dim)

    def forward(self, noise, x, t):
        step_embed = self.step_encoder(t)
        window = torch.cat([x, noise], dim=0)
        inp = self.pos_encoder(self.inp_fc(window)+step_embed.unsqueeze(0))
        out = self.TF(inp)
        return self.out_fc(out)[x.shape[0]:]


## codes referred from DLow's VAE code.
class GRUED_Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prefix_len = prefix_len
        self.pred_len = pred_len

        self.enc_gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers)
  
        self.dec_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers)

        self.step_encoder = DiffusionEmbedding(diff_steps, hidden_dim)

        self.inp_fc = nn.Linear(input_dim+hidden_dim, hidden_dim)
        self.init_h_fc = nn.Linear(hidden_dim, hidden_dim)
        self.init_y_fc = nn.Linear(hidden_dim, input_dim)
        self.out_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        feat, _ = self.enc_gru(x)
        return feat[-1]

    def decode(self, noise, feat, t):
        step_embed = self.step_encoder(t)

        h = self.init_h_fc(feat)

        rnn_in = torch.cat([noise, 
                            feat.unsqueeze(0).repeat(noise.shape[0], 1, 1)], dim=-1)
        rnn_in = self.inp_fc(rnn_in) + step_embed.unsqueeze(0)
        y, _ = self.dec_gru(rnn_in, h.unsqueeze(0).repeat(self.num_layers, 1, 1))
        res = self.out_fc(y)
        
        return res

    def forward(self, noise, x, t):
        return self.decode(noise, self.encode(x), t)

class GRU_Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, prefix_len, pred_len, diff_steps):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prefix_len = prefix_len
        self.pred_len = pred_len

        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers)
  
        self.step_encoder = DiffusionEmbedding(diff_steps, hidden_dim)

        self.inp_fc = nn.Linear(input_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, noise, x, t):
        step_embed = self.step_encoder(t)
        window = torch.cat([x, noise], dim=0)

        out, _ = self.gru(self.inp_fc(window)+step_embed)
        out = self.out_fc(out)
        
        return out[x.shape[0]:]