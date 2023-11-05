import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter
from hyperbolic.layers import HypLinear, HypAct

class StaticEmbedding(nn.Module):
    def __init__(self, stc_num, emb_dim):
        super(StaticEmbedding, self).__init__()
        self.stc_embed = nn.Parameter(torch.zeros((stc_num, emb_dim)))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.stc_embed)
    
    def forward(self):
        stc_embed = self.stc_embed
        return stc_embed

class Evolution(nn.Module):
    def __init__(self, emb_dim, rela_num):
        super(Evolution, self).__init__()
        self.rela_num = rela_num

        self.lstm = nn.LSTM(emb_dim, emb_dim, batch_first = True)
        self.rela_linears = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for r in range(rela_num)])

    def forward(self, embed, dyn_input):
        batch_sizes = dyn_input.batch_sizes
        stc_v, r = dyn_input.data.T
        seq_emb = embed[stc_v]

        for r_ in range(self.rela_num):
            linear = self.rela_linears[r_]
            idx = r == r_
            seq_emb[idx] = linear(seq_emb[idx])

        seq_emb_pack = PackedSequence(data = seq_emb, batch_sizes = batch_sizes)
        dyn_embed, _ = self.lstm(seq_emb_pack)
        dyn_embed = dyn_embed.data

        return dyn_embed

class Relation(nn.Module):
    def __init__(self, manifold, c, input_dim, emb_dim, rela_num):
        super(Relation, self).__init__()
        self.manifold = manifold
        self.c = c

        self.mlp = nn.Sequential(
            HypLinear(manifold, input_dim, emb_dim, c, dropout = 0.0, use_bias = True),
            HypAct(manifold, c, c, nn.Tanh()),
        )

        self.linear = nn.Linear(emb_dim, rela_num)
    
    def forward(self, edge_emb):
        rela = self.mlp(edge_emb)

        md = self.manifold
        c = self.c

        rela = md.logmap0(rela, c)
        rela = self.linear(rela)

        return rela

class GraphAttentionAggregationLayer(nn.Module):
    def __init__(self, manifold, c):
        super(GraphAttentionAggregationLayer, self).__init__()
        self.manifold = manifold
        self.c = c

    def forward(self, input, edge_index):
        input_size, input_dim = input.shape
        x_index, y_index = edge_index

        md = self.manifold
        c = self.c

        dist = md.sqdist(input[x_index], input[y_index], c)
        alpha = scatter_softmax(dist, index = x_index, dim = 0, dim_size = input_size)
        output = md.logmap0(input[y_index], c)
        output = alpha.unsqueeze(-1) * output
        output = scatter(output, index = x_index, dim = 0, dim_size = input_size, reduce = 'sum')
        output = md.expmap0(output, c)
        output = md.proj(output, c)

        return output

class GraphAttentionAggregation(nn.Module):
    def __init__(self, manifold, c, emb_dim, layer_num):
        super(GraphAttentionAggregation, self).__init__()
        self.manifold = manifold
        self.c = c

        GAALayer = GraphAttentionAggregationLayer
        self.gaa_layers = nn.ModuleList([GAALayer(manifold, c) for i in range(layer_num)])

        self.act = HypAct(manifold, c, c, nn.Tanh())
    
    def forward(self, input, edge_index):
        output = [input]
        hidden = input

        md = self.manifold
        c = self.c

        for gaa_layer in self.gaa_layers:
            hidden = gaa_layer(hidden, edge_index)
            hidden = self.act(hidden)
            hidden = md.proj(hidden, c)

            output.append(md.logmap0(hidden, c))
        
        output = torch.cat(output, dim = -1)
        output = md.expmap0(output, c)
        output = md.proj(output, c)

        return output
