import argparse

parser = argparse.ArgumentParser(add_help = False)

parser.add_argument('--quit_epoch_num', type = int, default = 10)

parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--device', type = str, default = 'cuda:0')

parser.add_argument('--data_path', type = str, default = '../dataset')
parser.add_argument('--data_name', type = str, default = 'assist2009')

parser.add_argument('--prop_seed', type = int, default = 0)
parser.add_argument('--trn_prop', type = float, default = 0.8)
parser.add_argument('--topk', type = int, nargs = '+', default = [1, 5])

parser.add_argument('--emb_dim', type = int, default = 128)
parser.add_argument('--batch_size', type = int, default = 2000)
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--weight_decay', type = float, default = 0.001)
parser.add_argument('--lamb', type = float, default = 0.1)
parser.add_argument('--neg_num', type = int, default = 8)
parser.add_argument('--margin', type = float, default = 1.)
parser.add_argument('--layer_num', type = int, default = 1)

parser.add_argument('--manifold', type = str, default = 'poincare')
parser.add_argument('--c', type = float, default = 1.)

parser.add_argument('--print_detail', action = 'store_true')
parser.add_argument('--print_result', action = 'store_true')

args, _ = parser.parse_known_args()