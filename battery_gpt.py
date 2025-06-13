import torch.nn as nn
from .fusion import MultiModalFusion
from .transformer import WindowedTransformer

class BatteryGPT(nn.Module):
    def __init__(self, input_dims, hidden_dim, n_heads, short_win, long_win):
        super(BatteryGPT, self).__init__()
        self.fusion = MultiModalFusion(input_dims, hidden_dim)
        self.transformer = WindowedTransformer(hidden_dim, hidden_dim, n_heads, short_win, long_win)
        self.regressor_soh = nn.Linear(hidden_dim, 1)
        self.regressor_rul = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        fused = self.fusion(inputs)
        trans_out = self.transformer(fused.unsqueeze(1))

        soh_pred = self.regressor_soh(trans_out)
        rul_pred = self.regressor_rul(trans_out)

        return soh_pred, rul_pred
