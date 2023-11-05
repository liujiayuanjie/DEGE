import torch
import warnings
from experiment import Experiment
from params import parser
from utils import Maximizer, Averager

class FileWriter:
    def __init__(self, path):
        self._path = path
        file = open(path, 'w')
        file.close()
    
    def __call__(self, content):
        file = open(self._path, 'a')
        file.write(content)
        file.close()
        return self

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    args, _ = parser.parse_known_args()

    experiment = Experiment(args)

    epoch = 1
    quit_count = 0

    rec_max = Maximizer()
    skl_atn_res = 0, 0, 0

    auc_max = Maximizer()
    pfm_prd_res = 0, 0

    dur_avg = Averager()

    while quit_count <= args.quit_epoch_num:
        edge_loss, rela_loss, skl_atn_results, pfm_prd_results, dur = experiment()
        auc, acc = pfm_prd_results
        topk, rec, pre, f1 = skl_atn_results[0]
        
        if rec_max(rec):
            skl_atn_res = skl_atn_results
            quit_count = 0
        
        if auc_max(auc):
            pfm_prd_res = pfm_prd_results
            quit_count = 0

        if args.print_detail:
            print('  '.join((
                'epoch: %-4d' % epoch,
                'edge_loss: %-.4f' % edge_loss,
                'rela_loss: %-.4f' % rela_loss,
                'rec: %-.4f/%-.4f' % (rec, rec_max.val),
                'pre: %-.4f' % pre,
                'auc: %-.4f/%-.4f' % (auc, auc_max.val),
                'dur: %-.2fs' % dur,
            )))
        
        dur_avg(dur)
        epoch += 1
        quit_count += 1

    if args.print_result:
        print('%.4f' % dur_avg.val)
        print('%.4f' % (int(torch.cuda.max_memory_allocated()) / 1024 ** 3))
        for topk, rec, pre, f1 in skl_atn_results:
            print('%.4f %.4f %.4f' % (rec, pre, f1))
        print('%.4f %.4f' % pfm_prd_res)