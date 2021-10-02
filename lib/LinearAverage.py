import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import math
from .alias_multinomial import AliasMethod


class LinearAverageOp(Function):
    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()
        batchSize = x.size(0)

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T)  # batchSize * N

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()

        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None


class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5,
                 sample_k=4096, topk=1):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.K = sample_k
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()

        self.register_buffer('params', torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

        # neighbour matrix indicating positive samples
        # a num_data length list of list storing positive index for each sample
        # e.g. [[0, 3, 5, 1], ...] means 0th, 1th, 3th, 5th are positive pairs
        self.affinity_mat = [set([int(i)]) for i in range(self.nLem)]
        self.topk = topk

    def forward(self, x, y):
        out = LinearAverageOp.apply(x, y, self.memory, self.params)
        return out

    def add_affinity(self, y, pos, r):
        """Add samples within R radius of pos into y's neighbour.
        y: [batch], index of image in the dataset
        pos: (synthetic) fake example, used for calculated neighbour, [batch, embedding_size]
        r: radius to determine neighbour, [batch]
            Actually I use cosine similarity here, so larger is more similar
            Places where r equals to -1 means not update neighbour for that batch
        """
        with torch.no_grad():
            batch_size = y.size(0)
            if batch_size == 0:
                return

            anchor_idx = [int(idx.item()) for idx in y]

            # avoid finding self!
            similarity = torch.mm(self.memory, pos.t()).t()  # [batch, num_data]
            for i in range(batch_size):
                similarity[i, anchor_idx[i]] = -1.  # set self sim to least

            topk_dist, topk_idx = \
                similarity.topk(self.topk, dim=1, largest=True, sorted=True)  # [batch, topk]

            # add affinity relationship to each sample
            for i in range(batch_size):
                add_idx = topk_idx[i][topk_dist[i] > r[i]]  # add those similar enough
                class_idx = anchor_idx[i]
                add_idx = [int(idx.item()) for idx in add_idx]
                for idx in add_idx:  # update affinity for both
                    self.affinity_mat[class_idx].add(idx)
                    self.affinity_mat[idx].add(class_idx)

    def sample_pos_neg(self, x, y):
        """Sample positive and negative according to y and its affinity."""
        with torch.no_grad():
            batchSize = x.size(0)
            # sample negative data idx
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)  # [batch, neg + 1]
            # make sure that positive are not included
            for i in range(batchSize):
                affinity = list(self.affinity_mat[int(y[i].item())])  # [num_neighbour]
                invalid = invalid_idx(idx[i], affinity)
                invalid_num = invalid.sum()
                while invalid_num > 0:
                    idx[i][invalid] = self.multinomial.draw(invalid_num).view(invalid_num)
                    invalid = invalid_idx(idx[i], affinity)
                    invalid_num = invalid.sum()

            # sample positive and negative samples
            idx.select(1, 0).copy_(y)  # set the first idx to be gt as positive sample
            # sample corresponding weights
            data_sample = torch.index_select(self.memory, 0, idx.view(-1)). \
                view(batchSize, self.K + 1, self.memory.size(1))
            pos_sample = data_sample[:, 0].data  # [batch, embedding_size]
            neg_sample = data_sample[:, 1:].data  # [batch, num_neg, embedding_size]
            selected_neg_sample = get_nearest_neg_sample(x, neg_sample)  # [batch, embedding_size]

        return pos_sample, selected_neg_sample


def get_nearest_neg_sample(anchor, neg):
    """Returns a negative sample per batch for G generating new triplet.
    Inputs:
        anchor: [batch, embedding_size]
        neg: [batch, num_neg, embedding_size]
    Returns:
        selected_neg: [batch, embedding_size]
    """
    dist = torch.bmm(neg, anchor.unsqueeze(-1)).squeeze(-1)  # [batch, num_neg]
    max_idx = torch.argmax(dist, dim=1)  # [batch]
    selected_neg = torch.stack([neg[i, max_idx[i]] for i in range(neg.size(0))], dim=0)
    return selected_neg


def invalid_idx(idx, affinity):
    """Return position index of positive sample in idx.
    Inputs:
        idx: [num_sample]
        affinity: a list of neighbours' idx
    Returns:
        invalid: [num_sample], 1 means this sample is invalid
    """
    invalid = (idx == affinity[0])
    for j in range(1, len(affinity)):
        invalid |= (idx == affinity[j])
    return invalid


class LinearCriterion(nn.Module):
    """Loss function for our GAN based method.
    We calculate instance for data without neighbour, and affinity loss for groups.
    """

    def __init__(self, hp_loss_weight=0., T=None):
        super(LinearCriterion, self).__init__()
        self.hp_loss_weight = hp_loss_weight
        self.T = T
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, index, aff_mat, memory=None):
        """logits: [batch, num_data]
        index: [batch], idx of img data
        aff_mat: a list of set storing neighbour relationships for each data
            len(aff_mat) == num_data
        """
        batch_size = index.size(0)
        preds = self.softmax(logits)
        aff_idx, inst_idx = self.__split(index, aff_mat)

        # calculate L_{inst}
        l_inst = torch.tensor(0.).cuda()
        if inst_idx.size(0) > 0:
            y_inst = index.index_select(0, inst_idx)
            x_inst = preds.index_select(0, inst_idx)
            # p_i = p_{i, i}
            x_inst = x_inst.gather(1, y_inst.view(-1, 1))
            # NLL: l = -log(p_i)
            l_inst = -1. * torch.log(x_inst).sum()

        # calculate L_{aff}
        l_aff = torch.tensor(0.).cuda()
        if aff_idx.size(0) > 0:
            y_aff = index.index_select(0, aff_idx)  # [num_aff]
            x_aff = preds.index_select(0, aff_idx)  # [num_aff, num_data]
            aff = [list(aff_mat[int(idx.item())]) for idx in y_aff]  # [num_aff, num_aff (including self)]
            batch_sum_p = torch.zeros((len(aff),)).float().cuda()
            for i in range(len(aff)):
                idx_aff = torch.from_numpy(np.array(aff[i])).long().cuda()  # [num_aff]
                sum_p = x_aff[i].index_select(0, idx_aff).sum()
                batch_sum_p[i] = sum_p
            l_aff = -1. * torch.log(batch_sum_p).sum()

        # calculate L_{hp}
        l_hp = torch.tensor(0.).cuda()
        if self.hp_loss_weight > 0. and aff_idx.size(0) > 0:
            l_hp = self.hard_pos_loss(preds.index_select(0, aff_idx),
                                      index.index_select(0, aff_idx),
                                      aff_mat, memory) * self.hp_loss_weight
        l_inst = l_inst / batch_size
        l_aff = l_aff / batch_size
        l_hp = l_hp / batch_size

        return l_inst + l_aff + l_hp, l_inst.item(), l_aff.item(), l_hp.item()

    def hard_pos_loss(self, probs, index, aff_mat, memory):
        """Calculate hard positive loss which the KLD between x_i and x_hp.
        probs: [batch_size, num_data], p_{i, j} (after softmax)
        index: [batch_size], idx of data x_i
        aff_mat: a list of set storing neighbour relationships for each data
            len(aff_mat) == num_data
        memory: [num_data, emb_size]
        """
        batch_size = probs.size(0)
        all_hp_logits = torch.zeros(probs.size()).cuda()
        for i in range(batch_size):
            # first get hard positive index
            idx = int(index[i].item())
            aff_idx = list(aff_mat[idx])
            aff_idx.remove(idx)
            fea_i = memory[idx]  # [emb_size]
            fea_aff = memory[aff_idx]  # [aff_num, emb_size]
            if fea_aff.size(0) == 1:  # only one pos
                fea_hp = fea_aff[0]  # [emb_size]
            else:
                sim = torch.mm(fea_aff, fea_i.unsqueeze(1)).squeeze(1)  # [aff_num]
                hp_idx = torch.argmax(sim, dim=0)
                fea_hp = fea_aff[hp_idx]  # [emb_size]
            # calculate p_{hp, j}
            with torch.no_grad():
                hp_logits = torch.mm(memory, fea_hp.unsqueeze(1)).squeeze(1) / self.T  # [num_data]
                all_hp_logits[i] = hp_logits  # pre-softmax
        # now we have probs p_{i, j} and hp_logits (before softmax)
        kld = torch.sum(probs * (torch.log(probs) -
                                 self.log_softmax(all_hp_logits)))
        return kld

    def __split(self, index, aff_mat):
        """Returns index with and without neighbour."""
        aff_num = torch.from_numpy(
            np.array([len(aff_mat[int(idx.item())]) for idx in index])).cuda()
        return (aff_num > 1).nonzero().view(-1), (aff_num <= 1).nonzero().view(-1)
