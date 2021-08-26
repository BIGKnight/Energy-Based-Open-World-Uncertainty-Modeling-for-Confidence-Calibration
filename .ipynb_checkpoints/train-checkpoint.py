import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
#import ipdb
import numpy as np
import wideresnet
import json
# Sampling
# from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3

def eval_ece(pred_scores_np, pred_np, label_np, num_bins=15):
    """Calculates ECE.

    Args:
      pred_scores_np: the softmax output at the dimension of the predicted
        labels of test samples.
      pred_np:  the numpy array of the predicted labels of test samples.
      label_np:  the numpy array of the ground-truth labels of test samples.
      num_bins: the number of bins to partition all samples. we set it as 15.

    Returns:
      ece: the calculated ECE value.
    """
    acc_tab = np.zeros(num_bins)  # Empirical (true) confidence
    mean_conf = np.zeros(num_bins)  # Predicted confidence
    nb_items_bin = np.zeros(num_bins)  # Number of items in the bins
    tau_tab = np.linspace(
        min(pred_scores_np), max(pred_scores_np),
        num_bins + 1)  # Confidence bins
    tau_tab = np.linspace(0, 1, num_bins + 1)  # Confidence bins
    for i in np.arange(num_bins):  # Iterates over the bins
        # Selects the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > pred_scores_np) & (pred_scores_np >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # Selects the predicted classes, and the true classes
        class_pred_sec, y_sec = pred_np[sec], label_np[sec]
        # Averages of the predicted max probabilities
        mean_conf[i] = np.mean(
            pred_scores_np[sec]) if nb_items_bin[i] > 0 else np.nan
        # Computes the empirical confidence
        acc_tab[i] = np.mean(
            class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]
    if sum(nb_items_bin) != 0:
        ece = np.average(
            np.absolute(mean_conf - acc_tab),
            weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
    else:
        ece = 0.0
    return ece

class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module):
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10, ebm_modify=False):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        if ebm_modify:
            self.class_output = nn.Linear(self.f.last_dim, n_classes + 1)
        else:
            self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None, in_embed=False, temp=1.):
        if in_embed == False:
            out = self.f.conv1(x) # [64, 16, 32, 32]
            out = self.f.layer1(out) # [64, 160, 32, 32]
            out = out + 0.01 * t.randn_like(out)
            out = self.f.layer2(out) # [64, 320, 16, 16]
            out = self.f.layer3(out) # [64, 640, 8, 8]
            out = self.f.lrelu(self.f.bn1(out))
        else:
#             out = self.f.conv1(x) # [64, 16, 32, 32]
#             out = self.f.layer1(out) # [64, 160, 32, 32]
#             out = self.f.layer2(x) # [64, 320, 16, 16]
            out = self.f.layer3(x)
            out = self.f.lrelu(self.f.bn1(out))
        if self.f.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)
        else:
            out = tnnF.avg_pool2d(out, 8)
        penult_z = out.view(out.size(0), -1)
    
        logits = self.class_output(penult_z) / temp
        return {
            'logits': logits, 
            'scores': t.softmax(logits, dim=1).squeeze()
        }

def cycle(loader):
    while True:
        for data in loader:
            yield data


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


def init_random(args, bs):
#     return t.FloatTensor(bs, args.width*64).uniform_(-0.1, 0.1)
#     return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)
    return t.FloatTensor(bs, 320, 16, 16).uniform_(-0.1, 0.1)
#     return t.FloatTensor(bs, 160, 32, 32).uniform_(-0.1, 0.1)
#     return t.randn(bs, 160, 32, 32) *x1


def get_model_and_buffer(args, device, sample_q):
#     model_cls = F if args.uncond else CCF
    model_cls = F
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes, ebm_modify=args.ebm_modify)
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None:
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size)
    else:
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path)
#         f.load_state_dict(ckpt_dict["model_state_dict"])
#         replay_buffer = ckpt_dict["replay_buffer"]
        f.load_state_dict(ckpt_dict)
        replay_buffer = init_random(args, args.buffer_size)

    f = f.to(device)
    return f, replay_buffer


def get_data(args):
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")

    # get all training inds
    full_train = dataset_fn(True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(1234)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds])
    if args.labels_per_class > 0:
        for i in range(args.n_classes):
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)
    dset_train_labeled = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_labeled_inds)
    dset_valid = DataSubset(
        dataset_fn(True, transform_test),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid, dload_test


def get_sample_q(args, device):
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(args, bs), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
        inds = t.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        inds = inds.long()
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(args, bs)
        choose_random = (t.rand(bs) < args.reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(f, replay_buffer, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs)
        embedding_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(1, n_steps +1):
            negative_energy = -t.log(f(embedding_k, in_embed=True)['scores'][:, -1] + 1e-12)
            f_prime = t.autograd.grad(negative_energy.sum(), [embedding_k], retain_graph=True)[0]
            embedding_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(embedding_k)
        f.train()
        final_samples = embedding_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q

def eval_classification(args, f, dload, device, temp=1.):
    corrects, losses = [], []
    scores_list, preds_list, labels_list = [], [], []
    with t.no_grad():
        for x_p_d, y_p_d in dload:
            x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
            output = f(x_p_d, temp=temp)
            logits = output['logits'][:, :args.n_classes]
    #         scores = tnnF.softmax(logits, dim=1)
            scores = output['scores'][:, :args.n_classes]

            scores_list.append(scores.max(dim=1)[0].cpu().numpy().reshape(-1))
            preds_list.append(scores.max(dim=1)[1].cpu().numpy().reshape(-1))
            labels_list.append(y_p_d.cpu().numpy().reshape(-1))

            loss = nn.CrossEntropyLoss(reduction='none')(logits, y_p_d).cpu().numpy()
            losses.extend(loss)
            correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
            corrects.extend(correct)
        
    scores_vec = np.reshape(scores_list, -1)
    preds_vec = np.reshape(preds_list, -1)
    labels_vec = np.reshape(labels_list, -1)
    ece = eval_ece(scores_vec, preds_vec, labels_vec, 15)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss, ece


def checkpoint(f, buffer, tag, args, device):
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)


def main(args):
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    sample_q = get_sample_q(args, device)
    f, replay_buffer = get_model_and_buffer(args, device, sample_q)

    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_test_acc = 0.0
    best_test_ece = -1
    cur_iter = 0

        
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * args.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))
        for i, (x_p_d, _) in enumerate(dload_train):
            if cur_iter <= args.warmup_iters:
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)
            L = 0.
            logits = f(x_lab)['logits']
            l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
            L += l_p_y_given_x
            acc = (logits.max(1)[1] == y_lab).float().mean()
            
            if args.ebm_modify and epoch >= args.ebm_start_epoch:
                embedding_q = sample_q(f, replay_buffer)
                y_q = t.ones(args.batch_size, device=device, dtype=t.long) * args.n_classes
                logits_q = f(embedding_q, in_embed=True)['logits']
                ebm_l = nn.CrossEntropyLoss()(logits_q, y_q) * args.ebm_weight 
                L += ebm_l
#                 print(ebm_l)
                    

            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print("BAD BOIIIIIIIIII")
                1/0

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1
            
        if epoch % args.eval_every == 0:
            ebm_loss = (ebm_l).item() if args.ebm_modify and epoch >= args.ebm_start_epoch else -1
            print('P(y|x) {}:{:>d} loss={:>.4f}, ebm_loss={:>.4f}, Train acc={:>.4f}'.format(epoch,
                                                                             cur_iter,
                                                                             l_p_y_given_x.item(),
                                                                             ebm_loss,
                                                                             acc.item()))
            f.eval()
            with t.no_grad():
                correct, loss, _ = eval_classification(args, f, dload_test, device)
                print("Epoch {}: Test Loss {:.4f}, Test Acc {:.4f}".format(epoch, loss, correct))
            f.train()
            
    for idx, (x_val, y_val) in enumerate(dload_valid):
        x_val, y_val = x_val.to(device), y_val.to(device)
        logits = f(x_val)['logits']
        loss_val = nn.CrossEntropyLoss()(logits, y_val)
        optim.zero_grad()
        loss_val.backward()
        optim.step()
        
    correct, loss, ece = eval_classification(args, f, dload_test, device)
    print("Epoch {}: Test Loss {:.4f}, Test Acc {:.4f}, Test ECE {:.4f}".format(epoch, loss, correct, ece))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[7, 10],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.5,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=100, help="number of steps of SGLD per iteration")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1)
    parser.add_argument("--sgld_std", type=float, default=1e-3)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000, help="the size of valid set, the plat schalling will be applied by using this valid set.")
    parser.add_argument('--gpu', type=int, default=0, help="gpus used in this experiments.")
    parser.add_argument('--ebm_modify', action='store_true', help="whether introduce the ebm objective")
    parser.add_argument('--ebm_start_epoch', type=int, default=0, help="start epoch of ebm training.")
    parser.add_argument('--ebm_weight', type=float, default=0.1, help="the loss weight of ebm part")
    

    args = parser.parse_args()
    args.n_classes = 100 if args.dataset == "cifar100" else 10
    t.cuda.set_device(args.gpu)
    main(args)