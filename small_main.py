'''Train ISL on small datasets with PyTorch.'''
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse
import time

import models
import datasets

from lib.LinearAverage import LinearAverage, LinearCriterion
from lib.utils import AverageMeter, set_seed, str2bool, get_lr, adjust_lr, \
    get_grad_norm, set_eval, set_train, stop_mining, stop_train_GAN, cal_aff_num
from test import kNN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
SEED = 1

set_seed(SEED)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100/SVHN Training')
parser.add_argument('--data', default='path-to-data', type=str, help='data root')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--arch', default='res18', type=str, help='network architecture')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', default=False, type=str2bool, help='test only')
parser.add_argument('--recompute', default=False, type=str2bool,
                    help='recompute features in testing')
parser.add_argument('--batch', default=128, type=int,
                    metavar='batch', help='batch_size for training')
parser.add_argument('--workers', default=8, type=int,
                    metavar='num_workers', help='num_workers for dataloader')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--sample_neg', default=4096, type=int,
                    help='sample neg to determine true_neg')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--eval_interval', default=5, type=int,
                    help='evaluation per interval epoch')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pos_score_threshold', default=0.5, type=float,
                    help='for adding neighbour')
parser.add_argument('--pos_aff_threshold', default=0.6, type=float,
                    help='affinity threshold for adding neighbour')
parser.add_argument('--hp_loss_weight', default=0.5, type=float,
                    help='loss weight for hard positive KLD')
parser.add_argument('--per_round_aff', default=0.5, type=float,
                    help='mined neighbour constraints per round')
parser.add_argument('--round', default=4, type=int,
                    help='total rounds to train GAN and model')
parser.add_argument('--backbone_epoch', default=200, type=int,
                    help='epoch num for training model')
parser.add_argument('--gan_epoch', default=200, type=int,
                    help='epoch num for training GAN')
parser.add_argument('--mining_epoch', default=200, type=int,
                    help='epoch num for mining affinity')
parser.add_argument('--topk', default=10, type=int,
                    help='select the topk nearest data as neighbour when mining')

args = parser.parse_args()

POS_SCORE_THRESHOLD = args.pos_score_threshold  # only D(fake_pos) > this will be used
R = args.pos_aff_threshold  # radius threshold for updating affinity
D_LOSS_WEIGHT = 1.
G_LOSS_WEIGHT = 1.
A_P_FAKE_P_LOSS_WEIGHT = 1.
A_FAKE_P_N_LOSS_WEIGHT = 1.
PER_ROUND_AFF = args.per_round_aff
HP_LOSS_WEIGHT = args.hp_loss_weight
TOPK = args.topk

device = 'cuda'
best_acc = 0.  # best test accuracy
start_round = 0
start_epoch = 0
next_stage = 0

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomGrayscale(p=0.2),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset.lower() == 'cifar10':
    dataset = datasets.CIFAR10Instance
elif args.dataset.lower() == 'cifar100':
    dataset = datasets.CIFAR100Instance
else:
    dataset = datasets.SVHNInstance

if args.dataset.lower() == 'svhn':
    trainset = dataset(root=args.data, split='train',
                       download=False, transform=transform_train)
    testset = dataset(root=args.data, split='test',
                      download=False, transform=transform_test)
else:
    trainset = dataset(root=args.data, train=True,
                       download=False, transform=transform_train)
    testset = dataset(root=args.data, train=False,
                      download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                          shuffle=True, num_workers=args.workers, pin_memory=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                         shuffle=False, num_workers=args.workers, pin_memory=True)

ndata = trainset.__len__()

print('==> Building model..')
if '18' in args.arch.lower():
    arch = 'cifar_resnet18'
elif '50' in args.arch.lower():
    arch = 'cifar_resnet50'
else:
    arch = 'cifar_alexnet'
net = models.__dict__[arch](low_dim=args.low_dim).cuda()
net = nn.DataParallel(net)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1,
                                                 factor=0.8, verbose=True, mode='max')
# GAN
# define GAN and loss function
generator = models.Generator(embedding_size=args.low_dim).cuda()
discriminator = models.Discriminator(embedding_size=args.low_dim).cuda()
gan_criterion = nn.BCELoss().cuda()
g_opt = optim.Adam(generator.parameters(), lr=1e-4)
d_opt = optim.Adam(discriminator.parameters(), lr=1e-4)

# define leminiscate
lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m,
                           sample_k=args.sample_neg, topk=TOPK).cuda()

cudnn.benchmark = True

# Model
if args.test_only or len(args.resume) > 0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    lemniscate.params = checkpoint['lemniscate'].params.cuda()
    lemniscate.memory = checkpoint['lemniscate'].memory.cuda()
    if hasattr(checkpoint['lemniscate'], 'affinity_mat'):
        lemniscate.affinity_mat = checkpoint['lemniscate'].affinity_mat
    best_acc = checkpoint['acc']
    if 'round' in checkpoint.keys():
        start_round = checkpoint['round']
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['opt'])
    if 'scheduler' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler'])
    if 'g_weight' in checkpoint.keys():
        generator.load_state_dict(checkpoint['g_weight'])
        discriminator.load_state_dict(checkpoint['d_weight'])
        g_opt.load_state_dict(checkpoint['g_opt'])
        d_opt.load_state_dict(checkpoint['d_opt'])
    if 'next_stage' in checkpoint.keys():
        next_stage = checkpoint['next_stage']

# define loss function
criterion = LinearCriterion(hp_loss_weight=HP_LOSS_WEIGHT, T=args.nce_t)

if args.test_only:
    acc = kNN(0, net, lemniscate, trainloader, testloader, 200,
              args.nce_t, recompute_memory=args.recompute)
    sys.exit(0)

# logs
start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logs_dir = os.path.join('logs', args.dataset, args.arch, str(start_datetime))
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)


# Training
def train_backbone(epoch):
    print('\nEpoch: %d' % epoch, 'train backbone')

    all_losses = AverageMeter()
    inst_losses = AverageMeter()
    aff_losses = AverageMeter()
    hp_losses = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    set_train(net)
    set_eval(generator)
    set_eval(discriminator)
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    end = time.time()
    for i, (inputs, targets, index) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets, index = inputs.cuda(), targets.cuda(), index.cuda()
        batch_size = index.size(0)

        # compute output
        feature = net(inputs)  # [batch, embedding_size]
        output = lemniscate(feature, index)  # [batch, num_data], p_{i, i}
        all_loss, inst_loss, aff_loss, hp_loss = criterion(output, index,
                                                           lemniscate.affinity_mat,
                                                           lemniscate.memory)

        # compute gradient and do SGD step
        all_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure accuracy and record loss
        all_losses.update(all_loss.item(), batch_size)
        inst_losses.update(inst_loss, batch_size)
        aff_losses.update(aff_loss, batch_size)
        hp_losses.update(hp_loss, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'Data {data_time.val:.3f}\t'
                  'InstLoss {inst_loss.val:.4f}\t'
                  'AffLoss {aff_loss.val:.4f}\t'
                  'HPLoss {hp_loss.val:.4f}\t'
                  'ALLLoss {all_loss.val:.4f}\t'.
                  format(epoch, i, len(trainloader),
                         batch_time=batch_time, data_time=data_time,
                         all_loss=all_losses,
                         aff_loss=aff_losses,
                         hp_loss=hp_losses,
                         inst_loss=inst_losses))
            torch.cuda.empty_cache()

    return all_losses.avg, inst_losses.avg, aff_losses.avg, hp_losses.avg


def train_GAN(epoch):
    print('\nEpoch: %d' % epoch, 'train GAN')

    data_time = AverageMeter()
    batch_time = AverageMeter()
    d_losses = AverageMeter()
    d_a_p_fake_p_losses = AverageMeter()
    d_a_fake_p_n_losses = AverageMeter()
    g_losses = AverageMeter()
    g_a_p_fake_p_losses = AverageMeter()
    g_a_fake_p_n_losses = AverageMeter()

    # will not train backbone in this stage
    set_eval(net)
    set_train(generator)
    set_train(discriminator)
    g_opt.zero_grad()
    d_opt.zero_grad()
    torch.cuda.empty_cache()

    end = time.time()
    for i, (inputs, targets, index) in enumerate(trainloader):
        data_time.update(time.time() - end)
        inputs, targets, index = inputs.cuda(), targets.cuda(), index.cuda()
        batch_size = index.size(0)

        g_opt.zero_grad()
        d_opt.zero_grad()

        with torch.no_grad():
            feature = net(inputs)  # [batch, embedding_size]

        # get positive and negative sample
        # both are [batch, embedding_size]
        true_pos, true_neg = lemniscate.sample_pos_neg(feature.data, index.data)

        #########################################
        # (1) Update D network
        #########################################
        set_train(discriminator)

        # get true triplet: [batch, 3 * embedding_size]
        with torch.no_grad():
            true_label = torch.ones((batch_size,)).float().cuda()
            fake_label = torch.zeros((batch_size,)).float().cuda()
            true_triplet = torch.cat([feature.data, true_pos.data, true_neg.data], dim=1)
        fake_pos = generator(true_triplet)  # [batch, embedding_size]
        a_p_fake_p = torch.cat([feature.data, true_pos.data, fake_pos], dim=1)
        a_fake_p_n = torch.cat([feature.data, fake_pos, true_neg.data], dim=1)

        # train D on fake triplet
        d_true_score = discriminator(true_triplet.data)
        d_true_loss = gan_criterion(d_true_score, true_label)
        a_p_fake_p_score = discriminator(a_p_fake_p.data)
        a_fake_p_n_score = discriminator(a_fake_p_n.data)
        a_p_fake_p_loss = gan_criterion(a_p_fake_p_score, fake_label) * \
                          A_P_FAKE_P_LOSS_WEIGHT / (A_P_FAKE_P_LOSS_WEIGHT + A_FAKE_P_N_LOSS_WEIGHT)
        a_fake_p_n_loss = gan_criterion(a_fake_p_n_score, fake_label) * \
                          A_FAKE_P_N_LOSS_WEIGHT / (A_P_FAKE_P_LOSS_WEIGHT + A_FAKE_P_N_LOSS_WEIGHT)
        d_fake_loss = a_p_fake_p_loss + a_fake_p_n_loss
        d_loss = (d_true_loss + d_fake_loss) * D_LOSS_WEIGHT / 2.
        d_a_p_fake_p_losses.update(a_p_fake_p_loss.item(), batch_size)
        d_a_fake_p_n_losses.update(a_fake_p_n_loss.item(), batch_size)
        d_losses.update(d_loss.item(), batch_size)

        d_loss.backward()
        d_opt.step()
        d_opt.zero_grad()

        #########################################
        # (2) Update G network
        #########################################
        set_eval(discriminator)

        a_p_fake_p_score = discriminator(a_p_fake_p)
        a_fake_p_n_score = discriminator(a_fake_p_n)
        a_p_fake_p_loss = gan_criterion(a_p_fake_p_score, true_label) * \
                          A_P_FAKE_P_LOSS_WEIGHT / (A_P_FAKE_P_LOSS_WEIGHT + A_FAKE_P_N_LOSS_WEIGHT)
        a_fake_p_n_loss = gan_criterion(a_fake_p_n_score, true_label) * \
                          A_FAKE_P_N_LOSS_WEIGHT / (A_P_FAKE_P_LOSS_WEIGHT + A_FAKE_P_N_LOSS_WEIGHT)
        fake_loss = a_p_fake_p_loss + a_fake_p_n_loss
        g_loss = fake_loss * G_LOSS_WEIGHT

        g_loss.backward()
        g_opt.step()
        g_opt.zero_grad()

        # update loss history
        g_losses.update(g_loss.item(), batch_size)
        g_a_p_fake_p_losses.update(a_p_fake_p_loss.item(), batch_size)
        g_a_fake_p_n_losses.update(a_fake_p_n_loss.item(), batch_size)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'
                  'Data {data_time.val:.3f}\t'
                  'GLoss {g_loss.val:.4f}\t'
                  'G_APP\'Loss {g_app_loss.val:.4f}\t'
                  'G_AP\'NLoss {g_apn_loss.val:.4f}\t'
                  'DLoss {d_loss.val:.4f}\t'
                  'D_APP\'Loss {d_app_loss.val:.4f}\t'
                  'D_AP\'NLoss {d_apn_loss.val:.4f}\t'.
                  format(epoch, i, len(trainloader),
                         batch_time=batch_time, data_time=data_time,
                         g_loss=g_losses,
                         g_app_loss=g_a_p_fake_p_losses,
                         g_apn_loss=g_a_fake_p_n_losses,
                         d_app_loss=d_a_p_fake_p_losses,
                         d_apn_loss=d_a_fake_p_n_losses,
                         d_loss=d_losses))
            torch.cuda.empty_cache()

    return d_losses.avg, g_losses.avg


def mine_affinity(epoch):
    print('\nEpoch: %d' % epoch, 'mine neighbours')

    # will not train anything in this stage
    set_eval(net)
    set_eval(generator)
    set_eval(discriminator)
    for i, (inputs, targets, index) in enumerate(trainloader):
        # generate fake_pos and get score for them
        inputs, targets, index = inputs.cuda(), targets.cuda(), index.cuda()
        feature = net(inputs)
        true_pos, true_neg = lemniscate.sample_pos_neg(feature.data, index.data)
        true_triplet = torch.cat([feature.data, true_pos.data, true_neg.data], dim=1)
        fake_pos = generator(true_triplet)  # [batch, embedding_size]

        # use [a, p', n] score as criterion
        a_fake_p_n = torch.cat([feature.data, fake_pos.data, true_neg.data], dim=1)
        a_fake_p_n_score = discriminator(a_fake_p_n).data
        d_fake_score = a_fake_p_n_score

        # mine new neighbour using fake_pos and radius
        # only use high confidence fake_pos, [batch]
        fake_pos_idx = (d_fake_score > POS_SCORE_THRESHOLD)
        r = R + (1. - d_fake_score) * (1. - R)  # [batch]
        r[~fake_pos_idx] = -1.  # -1 means this is true_pos

        # add affinity using fake_pos
        add_idx = ((r >= R) & (r <= 1.))  # [batch]
        lemniscate.add_affinity(index[add_idx], fake_pos[add_idx], r[add_idx])

        if i % args.print_freq == 0:
            aff_num = cal_aff_num(lemniscate.affinity_mat)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'nei_num: {3}\t'.
                  format(epoch, i, len(trainloader), aff_num))


if __name__ == '__main__':
    for round_idx in range(start_round, args.round):
        for epoch_idx in range(start_epoch, args.backbone_epoch):
            if round_idx == start_round and next_stage != 0:
                print('skip stage 0')
                break
            loss = train_backbone(epoch_idx)
            if (epoch_idx + 1) % args.eval_interval == 0:
                acc = kNN(epoch_idx, net, lemniscate, trainloader, testloader, 200, args.nce_t, False)
                scheduler.step(acc)

                print('Saving checkpoint after training backbone: Round {} Epoch {}'.
                      format(round_idx, epoch_idx))
                state = {
                    'net': net.state_dict(),
                    'lemniscate': lemniscate,
                    'acc': acc,
                    'round': round_idx,
                    'epoch': (epoch_idx + 1) % args.backbone_epoch,
                    'opt': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'g_weight': generator.state_dict(),
                    'd_weight': discriminator.state_dict(),
                    'g_opt': g_opt.state_dict(),
                    'd_opt': d_opt.state_dict(),
                    'next_stage': 1 if epoch_idx + 1 == args.backbone_epoch else 0
                }
                aff_num = cal_aff_num(lemniscate.affinity_mat)
                torch.save(state, os.path.join(logs_dir,
                                               'per_{}_model_{}-0-{}_acc_{}_'
                                               'all_{}_inst_{}_aff_{}_hp_{}_lr_{}.pth'.
                                               format(round(aff_num / ndata, 6),
                                                      round_idx + 1, epoch_idx + 1, round(acc, 4),
                                                      round(loss[0], 4), round(loss[1], 4),
                                                      round(loss[2], 4), round(loss[3], 4),
                                                      round(get_lr(optimizer), 6))))
                if acc > best_acc:
                    best_acc = acc

                print('best accuracy: {:.2f}'.format(best_acc * 100))
            start_epoch = 0
        if next_stage != 2:
            next_stage = 1
        losses = []
        for epoch_idx in range(start_epoch, args.gan_epoch):
            if round_idx == start_round and next_stage != 1:
                print('skip stage 1')
                break
            loss = train_GAN(epoch_idx)
            losses.append(loss[1])
            state = {
                'net': net.state_dict(),
                'lemniscate': lemniscate,
                'acc': best_acc,
                'round': round_idx,
                'epoch': (epoch_idx + 1) % args.gan_epoch,
                'opt': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'g_weight': generator.state_dict(),
                'd_weight': discriminator.state_dict(),
                'g_opt': g_opt.state_dict(),
                'd_opt': d_opt.state_dict(),
                'next_stage': 2 if epoch_idx + 1 == args.gan_epoch else 1
            }
            aff_num = cal_aff_num(lemniscate.affinity_mat)
            torch.save(state, os.path.join(logs_dir,
                                           'per_{}_model_{}-1-{}_'
                                           'd_{}_g_{}_lr_{}.pth'.
                                           format(round(aff_num / ndata, 6),
                                                  round_idx + 1, epoch_idx + 1,
                                                  round(loss[0], 4), round(loss[1], 4),
                                                  round(get_lr(optimizer), 6))))
            start_epoch = 0
            if stop_train_GAN(losses, is_round0=(round_idx == 0)):
                break
        for epoch_idx in range(start_epoch, args.mining_epoch):
            mine_affinity(epoch_idx)
            state = {
                'net': net.state_dict(),
                'lemniscate': lemniscate,
                'acc': best_acc,
                'round': round_idx + 1 if epoch_idx == args.mining_epoch else round_idx,
                'epoch': (epoch_idx + 1) % args.mining_epoch,
                'opt': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'g_weight': generator.state_dict(),
                'd_weight': discriminator.state_dict(),
                'g_opt': g_opt.state_dict(),
                'd_opt': d_opt.state_dict(),
                'next_stage': 0 if epoch_idx + 1 == args.mining_epoch else 2
            }
            aff_num = cal_aff_num(lemniscate.affinity_mat)
            torch.save(state, os.path.join(logs_dir,
                                           'num_{}_per_{}_model_{}-2-{}_lr_{}.pth'.
                                           format(aff_num,
                                                  round(aff_num / ndata, 6),
                                                  round_idx + 1, epoch_idx + 1,
                                                  round(get_lr(optimizer), 6))))
            if stop_mining(lemniscate.affinity_mat, round_idx, ndata, PER_ROUND_AFF):
                break
        start_epoch = 0

    acc = kNN(0, net, lemniscate, trainloader, testloader, 200, args.nce_t, True)
    print('last accuracy: {:.2f}'.format(acc * 100))
