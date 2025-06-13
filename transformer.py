import torch
import torch.nn as nn

class WindowedTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, short_win, long_win):
        super(WindowedTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_dim))
        self.short_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.long_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.short_win = short_win
        self.long_win = long_win
        self.scale_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]

        short_seq = x[:, -self.short_win:, :]
        long_seq = x[:, -self.long_win:, :]

        short_out, _ = self.short_attn(short_seq, short_seq, short_seq)
        long_out, _ = self.long_attn(long_seq, long_seq, long_seq)

        short_out = short_out.mean(dim=1)
        long_out = long_out.mean(dim=1)

        fused = self.scale_weight * short_out + (1 - self.scale_weight) * long_out
        output = self.ffn(fused)
        return output
