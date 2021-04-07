import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import util
from .stylegan2_layers import Downsample


def gan_loss(pred, should_be_classified_as_real):
    bs = pred.size(0)
    if should_be_classified_as_real:
        return F.softplus(-pred).view(bs, -1).mean(dim=1)
    else:
        return F.softplus(pred).view(bs, -1).mean(dim=1)


def feature_matching_loss(xs, ys, equal_weights=False, num_layers=6):
    loss = 0.0
    for i, (x, y) in enumerate(zip(xs[:num_layers], ys[:num_layers])):
        if equal_weights:
            weight = 1.0 / min(num_layers, len(xs))
        else:
            weight = 1 / (2 ** (min(num_layers, len(xs)) - i))
        loss = loss + (x - y).abs().flatten(1).mean(1) * weight
    return loss


class IntraImageNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query, target):
        num_locations = min(query.size(2) * query.size(3), self.opt.intraimage_num_locations)
        bs = query.size(0)
        patch_ids = torch.randperm(num_locations, device=query.device)

        query = query.flatten(2, 3)
        target = target.flatten(2, 3)

        # both query and target are of size B x C x N
        query = query[:, :, patch_ids]
        target = target[:, :, patch_ids]

        cosine_similarity = torch.bmm(query.transpose(1, 2), target)
        cosine_similarity = cosine_similarity.flatten(0, 1)
        target_label = torch.arange(num_locations, dtype=torch.long, device=query.device).repeat(bs)
        loss = self.cross_entropy_loss(cosine_similarity / 0.07, target_label)
        return loss


class VGG16Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_convs = torchvision.models.vgg16(pretrained=True).features
        self.register_buffer('mean',
                             torch.tensor([0.485, 0.456, 0.406])[None, :, None, None] - 0.5)
        self.register_buffer('stdev',
                             torch.tensor([0.229, 0.224, 0.225])[None, :, None, None] * 2)
        self.downsample = Downsample([1, 2, 1], factor=2)

    def copy_section(self, source, start, end):
        slice = torch.nn.Sequential()
        for i in range(start, end):
            slice.add_module(str(i), source[i])
        return slice

    def vgg_forward(self, x):
        x = (x - self.mean) / self.stdev
        features = []
        for name, layer in self.vgg_convs.named_children():
            if "MaxPool2d" == type(layer).__name__:
                features.append(x)
                if len(features) == 3:
                    break
                x = self.downsample(x)
            else:
                x = layer(x)
        return features

    def forward(self, x, y):
        y = y.detach()
        loss = 0
        weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0]
        #weights = [1] * 5
        total_weights = 0.0
        for i, (xf, yf) in enumerate(zip(self.vgg_forward(x), self.vgg_forward(y))):
            loss += F.l1_loss(xf, yf) * weights[i]
            total_weights += weights[i]
        return loss / total_weights


class NCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, query, target, negatives):
        query = util.normalize(query.flatten(1))
        target = util.normalize(target.flatten(1))
        negatives = util.normalize(negatives.flatten(1))
        bs = query.size(0)
        sim_pos = (query * target).sum(dim=1, keepdim=True)
        sim_neg = torch.mm(query, negatives.transpose(0, 1))
        all_similarity = torch.cat([sim_pos, sim_neg], axis=1) / 0.07
        #sim_target = util.compute_similarity_logit(query, target)
        #sim_target = torch.mm(query, target.transpose(0, 1)) / 0.07
        #sim_query = util.compute_similarity_logit(query, query)
        #util.set_diag_(sim_query, -20.0)

        #all_similarity = torch.cat([sim_target, sim_query], axis=1)

        #target_label = torch.arange(bs, dtype=torch.long,
        #                            device=query.device)
        target_label = torch.zeros(bs, dtype=torch.long, device=query.device)
        loss = self.cross_entropy_loss(all_similarity,
                                       target_label)
        return loss


class ScaleInvariantReconstructionLoss(nn.Module):
    def forward(self, query, target):
        query_flat = query.transpose(1, 3)
        target_flat = target.transpose(1, 3)
        dist = 1.0 - torch.bmm(
            query_flat[:, :, :, None, :].flatten(0, 2),
            target_flat[:, :, :, :, None].flatten(0, 2),
        )

        target_spatially_flat = target.flatten(1, 2)
        num_samples = min(target_spatially_flat.size(1), 64)
        random_indices = torch.randperm(num_samples, dtype=torch.long, device=target.device)
        randomly_sampled = target_spatially_flat[:, random_indices]
        random_indices = torch.randperm(num_samples, dtype=torch.long, device=target.device)
        another_random_sample = target_spatially_flat[:, random_indices]

        random_similarity = torch.bmm(
            randomly_sampled[:, :, None, :].flatten(0, 1),
            torch.flip(another_random_sample, [0])[:, :, :, None].flatten(0, 1)
        )

        return dist.mean() + random_similarity.clamp(min=0.0).mean()
