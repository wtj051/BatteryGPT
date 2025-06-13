import torch

def dynamic_masking(inputs, importance_weight=0.7):
    variability = torch.std(inputs, dim=1)
    gradient = torch.abs(inputs[:, 1:] - inputs[:, :-1]).mean(dim=1)
    importance_score = importance_weight * variability + (1 - importance_weight) * gradient
    exploration_score = 1 / torch.cumsum(torch.ones_like(inputs), dim=1)
    masking_prob = importance_score * exploration_score
    masking_prob = masking_prob / masking_prob.max()

    mask = torch.bernoulli(masking_prob).bool()
    masked_inputs = inputs.masked_fill(mask.unsqueeze(-1), 0)
    return masked_inputs, mask

