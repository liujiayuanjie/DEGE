import numpy as np
import torch
from utils import Divide
from torch.nn.utils.rnn import pack_sequence, PackedSequence

def read_csv(path):
    fp = open(path, 'r')
    rows = fp.read().split('\n')[: -1]
    split = lambda row: [int(e) if e.isdigit() else e for e in row.split(',')]
    rows = [split(row) for row in rows]

    return rows

class Data:
    def __init__(self, path, trn_prop, prop_seed = 0):
        # initialize vertexes
        vert_path = '%s/verts.csv' % path
        rows = read_csv(vert_path)
        rows = [row[1:] for row in rows]
        rows = np.array(rows).astype(np.int64)

        self.verts = verts = rows.T[0]
        self.types = types = rows.T[1]
        self.stc_num = stc_num = (types != 2).sum()

        self.qsts = qsts = verts[types == 0]
        self.skls = skls = verts[types == 1]

        # initialize edges
        edge_path = '%s/edges.csv' % path
        rows = read_csv(edge_path)
        edges = np.array(rows).astype(np.int64)[:, : -1]

        self.rela_num = edges[:, -1].max() + 1

        # divide questions
        qst_num = len(qsts)
        trn_num = int(qst_num * trn_prop)
        tst_num = qst_num - trn_num
        
        trn_qsts, tst_qsts = Divide(prop_seed)(qsts, [trn_num, tst_num])

        stc_edges = edges[edges[:, -1] > 1]

        qsts_ = stc_edges[:, 0]
        idx = np.isin(qsts_, trn_qsts)

        self.stc_edges = stc_edges = stc_edges[idx]

        qst_skls = {}
        for qst, skl, r in edges[edges[:, -1] == 2]:
            if qst not in qst_skls:
                qst_skls[qst] = set()
            qst_skls[qst].add(skl)

        self.trn_qsts = trn_qsts
        self.tst_qsts = tst_qsts
        self.qst_skls = qst_skls

        # initialize sequences
        usrs = verts[types == 2]
        seqs = [[] for usr in usrs]
        min_usr = usrs.min()

        for usr, qst, cor in edges:
            if cor < 2: seqs[usr - min_usr].append([qst, cor])
        
        idx = stc_num
        
        seq_data = []

        for seq in seqs:
            data = []
            for curr, next in zip(seq[: -1], seq[1:]):
                data.append([*curr, *next, 1])
            data[-1][-1] = 0
            seq_data.append(torch.tensor(data))
        
        seq_pack = pack_sequence(seq_data, enforce_sorted = False)
        seq_data = seq_pack.data
        batch_sizes = seq_pack.batch_sizes

        dyn_input = PackedSequence(data = seq_data[:, [0, 1]], batch_sizes = batch_sizes)
        self.dyn_input = dyn_input

        dyn_len = seq_data.size(0)
        usrs = torch.arange(stc_num, stc_num + dyn_len).unsqueeze(-1)
        dyn_edges = torch.cat((usrs, seq_data[:, [2, 3]]), dim = -1)
        dyn_edges = dyn_edges.numpy()

        self.v_num = v_num = stc_num + dyn_len

        dyn_edge_mask = seq_data[:, 4].numpy()

        cat = np.concatenate

        self.trn_edges = cat((stc_edges, dyn_edges[dyn_edge_mask == 1]), axis = 0)
        self.tst_edges = dyn_edges[dyn_edge_mask == 0]

        edge_idx1 = cat((stc_edges[:, [0, 1]], dyn_edges[:, [0, 1]]), axis = 0)
        edge_idx2 = cat((stc_edges[:, [1, 0]], dyn_edges[dyn_edge_mask == 1][:, [1, 0]]), axis = 0)
        edge_idx3 = np.arange(10).repeat(2).reshape(-1, 2)

        edge_idx = cat((edge_idx1, edge_idx2, edge_idx3), axis = 0)
        self.edge_idx = edge_idx.T