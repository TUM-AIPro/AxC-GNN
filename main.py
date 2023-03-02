import argparse
import sys
import os
from train_eval import graph_classification, node_classification
import torch_geometric.transforms as T

sys.path.append("code/python/")
from dataset import get_dataset
from nets import GIN, CitGIN, SAGE, CitSAGE

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--runs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--hidden", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=1e-6)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)
parser.add_argument("--lr_decay_step_size", type=int, default=50)
parser.add_argument("--path", type=str, default="../data/", help="dataset location")
parser.add_argument("--low", type=float, default=0.0)
parser.add_argument("--change", type=float, default=0.1)
parser.add_argument("--sample_prop", type=float, default=None)
parser.add_argument("--qbits", type=int, default=32)
parser.add_argument("--apx", type=int, default=0)
parser.add_argument("--trainapx", type=int, default=0)
parser.add_argument("--net", type=str, default='GIN')
parser.add_argument("--qtype", type=str, default='INT')
parser.add_argument("--dataset", type=str, default='PROTEINS')
parser.add_argument("--evalonly", action="store_true", help="eval only mode")
parser.add_argument("--poly", action="store_true", help="enable Poly scheduler")
parser.add_argument("--qinput", action="store_false", help="disables quantization in input")
parser.add_argument("--qweight", action="store_false", help="disables quantization in weights")
parser.add_argument("--qfeatures", action="store_false", help="disables quantization in features")

args = parser.parse_args()

dataset_name = args.dataset

if args.qbits == 32:
    qtype = "FP32"
    args.trainapx = 0
    DQ = None
else:
    qtype = args.qtype + str(args.qbits)
    DQ = {"prob_mask_low": args.low, "prob_mask_change": args.change}

ste = False
momentum = False
percentile = 0.001

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
run_citation = args.dataset=='Cora' or args.dataset=='CiteSeer' or args.dataset=='PubMed'
dataset = get_dataset(args.path, args.dataset, sparse=True, DQ=DQ)
data = dataset[0]

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
else:
    raise NotImplementedError

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
