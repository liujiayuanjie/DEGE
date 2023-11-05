import torch
import torch.nn.functional as F
import numpy as np
from utils import Device, Batcher, Averager, Timer
from utils import set_seed, cal_acc, cal_auc
from data import Data
from model import StaticEmbedding, Evolution, Relation, GraphAttentionAggregation
from hyperbolic.manifolds import PoincareBall
from hyperbolic.optimizers import RiemannianAdam

class Experiment:
    def __init__(self, args):
        # initialize basic settings
        self.args = args
        self.dev = dev = Device(args.device)
        set_seed(args.seed)

        self.manifold = manifold = PoincareBall()
        self.c = c = args.c

        # initialize data
        path = '%s/%s' % (args.data_path, args.data_name)
        self.data = data = Data(path, args.trn_prop, args.prop_seed)
        self.dyn_input = dev.attach(self.data.dyn_input)
        self.edge_idx = dev.tensor(self.data.edge_idx)

        # initialize models
        stc_num = data.stc_num
        emb_dim = args.emb_dim
        rela_num = data.rela_num
        layer_num = args.layer_num
        
        self.embed = embed = dev.attach(StaticEmbedding(stc_num, emb_dim))
        self.evol = evol = dev.attach(Evolution(emb_dim, rela_num))
        self.rela = rela = dev.attach(Relation(manifold, c, emb_dim * (layer_num + 1) * 2, emb_dim, rela_num))
        self.gaa = gaa = dev.attach(GraphAttentionAggregation(manifold, c, emb_dim, layer_num))

        embed_params = {'params': embed.parameters(), 'weight_decay': args.weight_decay}
        evol_params = {'params': evol.parameters()}
        rela_params = {'params': rela.parameters()}
        gaa_params = {'params': gaa.parameters()}

        parameters = [embed_params, evol_params, rela_params, gaa_params]
        self.optimizer = torch.optim.Adam(parameters, lr = args.lr)
        # self.optimizer = RiemannianAdam(parameters, lr = args.lr)
        self.criterion = torch.nn.CrossEntropyLoss()

        # initialize sampling probability
        edge_verts = data.trn_edges[:, : -1]
        probs = [0.0] * data.v_num

        for vert in edge_verts.reshape(-1):
            probs[vert] += 1
        
        probs = np.array(probs)
        self.probs = probs / probs.sum()

    def get_v_emb(self):
        stc_emb = self.embed()
        dyn_emb = self.evol(stc_emb, self.dyn_input)
        v_emb = torch.cat((stc_emb, dyn_emb), dim = 0)

        md = self.manifold
        c = self.c

        v_emb = md.expmap0(v_emb, c)
        v_emb = md.proj(v_emb, c)

        gaa_emb = self.gaa(v_emb, self.edge_idx)

        return v_emb, gaa_emb

    def evaluate(self):
        args = self.args
        dev = self.dev
        data = self.data

        self.embed.eval()
        self.evol.eval()
        self.rela.eval()
        
        embed, gaa_emb = self.get_v_emb()

        md = self.manifold
        c = self.c       

        # evaluate skill annotation
        qsts = data.tst_qsts

        qst_emb = embed[qsts]
        skl_emb = embed[data.skls]

        dist = md.mut_sqdist(qst_emb, skl_emb, c)
        sim_mat = -dist

        sim_mat = torch.mm(qst_emb, skl_emb.T)
        sim_mat = F.sigmoid(sim_mat)

        skl_atn_results = []

        for topk in args.topk:
            rec_avg, pre_avg, f1_avg = Averager(), Averager(), Averager()
            qst_skls = torch.topk(sim_mat, topk, dim = 1).indices + data.skls.min()
        
            for i in range(len(qsts)):
                qst = qsts[i]
                skls = data.qst_skls[qst]
                topk_skls = qst_skls[i]
                
                tp = len(set(dev.tolist(topk_skls)) & skls)
                rec = tp / len(skls)
                pre = tp / topk
                f1 = 2 * rec * pre / (rec + pre) if rec + pre != 0 else 0

                rec_avg(rec)
                pre_avg(pre)
                f1_avg(f1)

            skl_atn_results.append([topk, rec_avg.val, pre_avg.val, f1_avg.val])
        
        # evaluate performance prediction
        v1, v2, r = dev.tensor(data.tst_edges).T
        
        v1_emb = gaa_emb[v1]
        v2_emb = gaa_emb[v2]


        emb_cat = torch.cat((md.logmap0(v1_emb, c), md.logmap0(v2_emb, c)), dim = -1)
        emb_cat = md.expmap0(emb_cat, c)

        p = self.rela(emb_cat)

        p = p[:, : 2]
        p = p.softmax(dim = -1)[:, 1]

        p = p.squeeze(-1)
        p = dev.numpy(p)
        r = dev.numpy(r.float())
        
        auc = cal_auc(r, p)
        acc = cal_acc(r, p)

        pfm_prd_results = auc, acc

        return skl_atn_results, pfm_prd_results

    def train(self):
        args = self.args
        dev = self.dev

        self.embed.train()
        self.evol.train()
        self.rela.train()

        edge_loss_avg = Averager()
        edge_batcher = Batcher(self.data.trn_edges, args.batch_size)

        rela_loss_avg = Averager()
        rela_batcher = Batcher(self.data.trn_edges, args.batch_size)

        for edge_batch, rela_batch in zip(edge_batcher, rela_batcher):
            embed, gaa_emb = self.get_v_emb()

            edge_batch = dev.tensor(edge_batch)
            rela_batch = dev.tensor(rela_batch)

            edge_loss = self.get_edge_loss(embed, edge_batch)
            rela_loss = self.get_rela_loss(gaa_emb, rela_batch)

            edge_loss_avg(*dev.tolist(edge_loss))
            rela_loss_avg(dev.tolist(rela_loss))

            edge_loss = edge_loss.sum()
            loss = edge_loss + args.lamb * rela_loss * edge_batch.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return edge_loss_avg.val, rela_loss_avg.val
    
    def get_edge_loss(self, embed, g_batch):
        v1, v2, r = g_batch.T
        neg_num = self.args.neg_num
        
        v1_emb = embed[v1]
        v2_emb = embed[v2]

        v1_negs = self.neg_sample((*v1.size(), neg_num))
        v2_negs = self.neg_sample((*v2.size(), neg_num))

        v1_neg_emb = embed[v1_negs]
        v2_neg_emb = embed[v2_negs]

        md = self.manifold
        c = self.c

        pos = md.sqdist(v1_emb, v2_emb, c) ** 0.5
        neg1 = md.sqdist(v1_emb.unsqueeze(1), v1_neg_emb, c) ** 0.5
        neg2 = md.sqdist(v2_emb.unsqueeze(1), v2_neg_emb, c) ** 0.5

        margin = self.args.margin
        pos = pos.unsqueeze(-1)
        edge_loss = F.relu(pos - neg1 + margin) + F.relu(pos - neg2 + margin)
        edge_loss = edge_loss.mean(-1)

        return edge_loss
    
    def get_rela_loss(self, gaa_emb, p_batch):
        v1, v2, r = p_batch.T
        
        v1_emb = gaa_emb[v1]
        v2_emb = gaa_emb[v2]

        md = self.manifold
        c = self.c

        emb_cat = torch.cat((md.logmap0(v1_emb, c), md.logmap0(v2_emb, c)), dim = -1)
        emb_cat = md.expmap0(emb_cat, c)

        p = self.rela(emb_cat)
        rela_loss = self.criterion(p.squeeze(-1), r)
        
        return rela_loss

    def neg_sample(self, size):
        sample = np.random.choice(
            self.data.v_num, 
            size = size, 
            replace = True, 
            p = self.probs
        )
        sample = self.dev.tensor(sample).long()

        return sample
    
    def __call__(self):
        timer = Timer()

        edge_loss, rela_loss = timer(self.train)
        skl_atn_results, pfm_prd_results = timer(self.evaluate)

        return edge_loss, rela_loss, skl_atn_results, pfm_prd_results, timer.dur