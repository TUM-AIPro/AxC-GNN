import argparse
import sys
import os
from train_eval import graph_classification, node_classification
import torch_geometric.transforms as T

sys.path.append("code/python/")
from dataset import get_dataset
from nets import GIN, CitGIN, SAGE, CitSAGE, GAT, CitGAT, GCN, CitGCN

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100, help="epochs per one run")
parser.add_argument("--runs", type=int, default=1, help="how many times to run training, best accuracy will be selected from all runs")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--num_layers", type=int, default=3, help="number of hidden GNN layers")
parser.add_argument("--hidden", type=int, default=64, help="size of hidden GNN layers")
parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument("--lr_decay_factor", type=float, default=0.9, help="learning rate decay factor for Step scheduler")
parser.add_argument("--lr_decay_step_size", type=int, default=25, help="step size decay factor for Step scheduler")
parser.add_argument("--path", type=str, default="../data/", help="dataset location")
parser.add_argument("--low", type=float, default=0.0, help="minimum quantization probability for DegreeQuant")
parser.add_argument("--change", type=float, default=0.1, help="--low + --change - maximum quantization probability for DegreeQuant")
parser.add_argument("--sample_prop", type=float, default=None, help="allows to use sampling on tensors before running the percentile operation, not required")
parser.add_argument("--qbits", type=int, default=32, help="number of bits used for quantization, used and tested: 32 (FP32), 8-4 (int quantization)")
parser.add_argument("--apx", type=int, default=0, help="approximation mode: 0-precise, 1-1KV6, 2-1KV8, 3-1KV9, 4-1KVP, 5-1L2J, 6-1L2L, 7-1L2N, 8-1L12")
parser.add_argument("--trainapx", type=int, default=0, help="enables approximation-aware training, specify approximation mode to train with (as per --apx), overrides --apx if not in --evalonly mode")
parser.add_argument("--net", type=str, default='GIN', help="GIN, SAGE, GAT, GCN (case sensitive)")
parser.add_argument("--qtype", type=str, default='INT', help="only INT")
parser.add_argument("--dataset", type=str, default='MUTAG', help='MUTAG, ENZYMES, PROTEINS, COLLAB, IMDB-BINARY, REDDIT-BINARY, Cora, CiteSeer, PubMed (case sensitive)')
parser.add_argument("--evalonly", action="store_true", help="evaluation only mode, different --trainapx and --apx show how model trained with --trainapx QAT performes with --apx approximation")
parser.add_argument("--poly", action="store_true", help="enable Polynomial scheduler, replaces Step")
parser.add_argument("--qinput", action="store_false", help="disables quantization in inputs")
parser.add_argument("--qweight", action="store_false", help="disables quantization in weights")
parser.add_argument("--qfeatures", action="store_false", help="disables quantization in features")

args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")

# set-up quantization type and DegreeQuant
if args.qbits == 32:
    qtype = "FP32"
    args.trainapx = 0
    DQ = None
else:
    qtype = args.qtype + str(args.qbits)
    DQ = {"prob_mask_low": args.low, "prob_mask_change": args.change}

# quantization parameters, selection below demonstrated better results for DegreeQuant
ste = False
momentum = False
percentile = 0.001

# save/load path
if args.trainapx > 0:
    outdir = os.path.join("models", args.dataset, args.net, qtype, 'apx'+str(args.trainapx))
    if not args.evalonly:
        args.apx = args.trainapx
else:
    outdir = os.path.join("models", args.dataset, args.net, qtype)

if not os.path.exists(outdir):
    os.makedirs(outdir)

print('-----------------------------------------')
if not args.evalonly:
    print(args)
run_citation = args.dataset=='Cora' or args.dataset=='CiteSeer' or args.dataset=='PubMed' # set a True/False flag to select a proper GNN/task later

# load dataset
dataset = get_dataset(args.path, args.dataset, sparse=True, DQ=DQ)
data = dataset[0]

# load GNN
if args.net == 'GIN':
    if run_citation:
        model = CitGIN(
            dataset,
            hidden=args.hidden,
            qtype=qtype,
            qbits=args.qbits,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=args.sample_prop,
            q_input=args.qinput,
            q_weight=args.qweight,
            q_features=args.qfeatures,
            apx=args.apx,
        )
    else:
        model = GIN(
            dataset,
            num_layers=args.num_layers,
            hidden=args.hidden,
            qtype=qtype,
            qbits=args.qbits,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=args.sample_prop,
            q_input=args.qinput,
            q_weight=args.qweight,
            q_features=args.qfeatures,
            apx=args.apx,
        )
elif args.net == 'SAGE':
    if run_citation:
        model = CitSAGE(
            dataset,
            hidden=args.hidden,
            qtype=qtype,
            qbits=args.qbits,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=args.sample_prop,
            q_input=args.qinput,
            q_weight=args.qweight,
            q_features=args.qfeatures,
            apx=args.apx,
        )
    else:
        model = SAGE(
            dataset,
            num_layers=args.num_layers,
            hidden=args.hidden,
            qtype=qtype,
            qbits=args.qbits,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=args.sample_prop,
            q_input=args.qinput,
            q_weight=args.qweight,
            q_features=args.qfeatures,
            apx=args.apx,
        )
elif args.net == 'GAT':
    if run_citation:
        model = CitGAT(
            dataset,
            hidden=args.hidden,
            qtype=qtype,
            qbits=args.qbits,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=args.sample_prop,
            q_input=args.qinput,
            q_weight=args.qweight,
            q_features=args.qfeatures,
            apx=args.apx,
        )
    else:
        model = GAT(
            dataset,
            num_layers=args.num_layers,
            hidden=args.hidden,
            qtype=qtype,
            qbits=args.qbits,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=args.sample_prop,
            q_input=args.qinput,
            q_weight=args.qweight,
            q_features=args.qfeatures,
            apx=args.apx,
        )
elif args.net == 'GCN':
    if run_citation:
        model = CitGCN(
            dataset,
            hidden=args.hidden,
            qtype=qtype,
            qbits=args.qbits,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=args.sample_prop,
            q_input=args.qinput,
            q_weight=args.qweight,
            q_features=args.qfeatures,
            apx=args.apx,
        )
    else:
        model = GCN(
            dataset,
            num_layers=args.num_layers,
            hidden=args.hidden,
            qtype=qtype,
            qbits=args.qbits,
            ste=ste,
            momentum=momentum,
            percentile=percentile,
            sample_prop=args.sample_prop,
            q_input=args.qinput,
            q_weight=args.qweight,
            q_features=args.qfeatures,
            apx=args.apx,
        )
else:
    raise NotImplementedError("Supported GNN architectures: GIN, SAGE, GAT, GCN (case sensitive)")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Num. parameters: ', count_parameters(model))

# run training/evaluation
if args.apx > 0:
    print('Approximate mode #: ', args.apx)
if run_citation:
    acc = node_classification(
        model,
        data,
        args,
        outdir=outdir,
    )
    print("Result - {}".format(acc))
else:
    acc = graph_classification(
        dataset,
        model,
        args,
        folds=10,
        outdir=outdir,
    )

    desc = "{:.3f}".format(acc)
    print("Result - {}".format(desc))
print('-----------------------------------------')
