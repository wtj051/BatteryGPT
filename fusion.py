import torch
import torch.nn as nn

class MultiModalFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim):
        super(MultiModalFusion, self).__init__()
        self.encoders = nn.ModuleList([
            nn.GRU(input_size=dim, hidden_size=hidden_dim, batch_first=True)
            for dim in input_dims
        ])
        self.attention_vector = nn.Parameter(torch.randn(hidden_dim))
        self.modality_weights = nn.Parameter(torch.randn(len(input_dims)))

    def forward(self, inputs):
        modality_features = []
        for encoder, inp in zip(self.encoders, inputs):
            out, _ = encoder(inp)
            modality_features.append(out)

        modality_attentions = []
        for features in modality_features:
            score = torch.matmul(features, self.attention_vector)
            attention = torch.softmax(score, dim=1)
            weighted_features = (features * attention.unsqueeze(-1)).sum(dim=1)
            modality_attentions.append(weighted_features)

        modality_attentions = torch.stack(modality_attentions, dim=1)
        weights = torch.softmax(self.modality_weights, dim=0).unsqueeze(0)
        fused_output = (modality_attentions * weights).sum(dim=1)

        return fused_output
