import torch
import time
from lib.utils import AverageMeter


def kNN(epoch, net, lemniscate, trainloader, testloader, K, sigma, recompute_memory=False):
    with torch.no_grad():
        net.eval()
        net_time = AverageMeter()
        cls_time = AverageMeter()
        total = 0
        testsize = testloader.dataset.__len__()

        trainFeatures = lemniscate.memory.t()
        if hasattr(trainloader.dataset, 'imgs'):
            trainLabels = torch.LongTensor([y for (p, y) in trainloader.dataset.imgs]).cuda()
        else:
            trainLabels = torch.LongTensor(trainloader.dataset.train_labels).cuda()
        C = trainLabels.max() + 1

        if recompute_memory:
            transform_bak = trainloader.dataset.transform
            trainloader.dataset.transform = testloader.dataset.transform
            temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=256,
                                                     shuffle=False, num_workers=16)
            for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)
                batchSize = inputs.size(0)
                features = net(inputs)
                trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()
            trainloader.dataset.transform = transform_bak

        top1 = 0.
        top5 = 0.
        end = time.time()
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            inputs = inputs.cuda(async=True)
            targets = targets.cuda(async=True)
            batchSize = inputs.size(0)
            features = net(inputs)
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C),
                                        yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()

            total += targets.size(0)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

    print(top1 * 100. / total)

    return top1 / total
