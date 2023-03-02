import time
from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader, DenseDataLoader as DenseLoader
from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-5):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def node_classification(
    model,
    data,
    args,
    outdir=None,
    use_tqdm=True,
):
    best_acc = 0.
    model.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    data.to(device)
    filename = os.path.join(outdir, 'model.tar')
    if os.path.isfile(filename) and args.evalonly:
        use_tqdm = False
        args.epochs = 1
        args.runs=1
    for run_num in range(args.runs):
        if use_tqdm:
            t = tqdm(total=args.epochs, desc="Run #" + str(run_num+1))
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if args.poly:
            scheduler = PolyLR(optimizer, args.epochs, power=0.9)
        else:
            scheduler = StepLR(optimizer,args.lr_decay_step_size, args.lr_decay_factor)
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            train_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            train_loss.backward()
            optimizer.step()

            train_acc, val_acc, test_acc = test_planetoid(model, data)

            scheduler.step()
            if use_tqdm:
                t.set_postfix(
                    {
                        "Train_Loss": "{:05.3f}".format(train_loss),
                        "Test_Acc": "{:05.3f}".format(test_acc),
                    }
                )
                t.update(1)

            if test_acc > best_acc and not args.evalonly:
                best_acc = test_acc
                torch.save({
                'state_dict': model.state_dict(),
                'best_prec': best_acc,
                }, filename)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if os.path.isfile(filename) and args.evalonly:
        print("Running test on: ", str(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        _, _, test_acc = test_planetoid(model, data)
        return test_acc

    return best_acc


@torch.no_grad()
def test_planetoid(model, data):
    model.eval()
    pred = model(data).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def graph_classification(
    dataset,
    model,
    args,
    folds,
    use_tqdm=True,
    outdir=None,
):
    best_acc = 0.0
    filename = os.path.join(outdir, 'model.tar')
    if os.path.isfile(filename) and args.evalonly:
        use_tqdm = False
        args.epochs = 1
        args.runs=1

    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if "adj" in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, args.batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, args.batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, args.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

        model.to(device).reset_parameters()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for run_num in range(args.runs):
            model.reset_parameters()
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            if args.poly:
                scheduler = PolyLR(optimizer, args.epochs, power=0.9)
            else:
                scheduler = StepLR(optimizer,args.lr_decay_step_size, args.lr_decay_factor)
            if use_tqdm:
                t = tqdm(total=args.epochs, desc="Run #" + str(run_num+1))
            for epoch in range(1, args.epochs + 1):
                train_loss = train(model, optimizer, train_loader)
                epoch_acc = eval_acc(model, test_loader)

                scheduler.step()

                if epoch_acc > best_acc and not args.evalonly:
                    best_acc = epoch_acc
                    torch.save({
                        'state_dict': model.state_dict(),
                        'best_prec': best_acc,
                        }, os.path.join(filename))

                if use_tqdm:
                    t.set_postfix(
                        {
                            "Train_Loss": "{:05.3f}".format(train_loss),
                            "Val_Acc": "{:05.3f}".format(epoch_acc),
                        }
                    )
                    t.update(1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        break

    if os.path.isfile(filename) and args.evalonly:
        print("Running test on: ", str(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        best_acc = eval_acc(model, test_loader)

    return best_acc


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=2022)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)
