import torch
import torch.nn as nn
import torch.nn.functional as F



def getting_pic(predicted_labels, target, criterion):
    cross_entropy_loss = criterion(predicted_labels, target).squeeze()
    cross_entropy_loss = (-1) * cross_entropy_loss
    p_i_c = torch.exp(cross_entropy_loss)
    return p_i_c


def opposite_loss(predicted_labels, predicted_hardness_scores, target, criterion):
    # predicted_labels = F.softmax(predicted_labels, dim=1)
    cross_entropy_loss = criterion(predicted_labels, target).squeeze()
    cross_entropy_loss = (-1) * cross_entropy_loss
    p_i_c = torch.exp(cross_entropy_loss)
    # p_i_c = predicted_labels[0:, target.squeeze()]

    term1 = (1 - p_i_c) * predicted_hardness_scores
    term2 = (1 - predicted_hardness_scores) * p_i_c
    final_loss = 1 - term1 - term2
    return torch.mean(final_loss)

