import torch.nn as nn

class GatedDualHeadMLP(nn.Module):
    """
    Gated dual-head architecture for biocharge delta prediction.

    Mirrors the analytical model's hard if/else branching on sleep_state:
    - Awake head: learns stress-driven depletion, micro-recovery, HRR/ACC expenditure
    - Sleep head: learns stage/quality/HRV-modulated exponential recovery
    - Hard gate on sleep_markers selects which head's output to use
    - Shared encoder projects all features into a common representation

    Uses SiLU (Swish) activation to match the analytical model's exponential formulas.
    ~10k parameters with default dims (shared=48, awake=64/32, sleep=48/24).
    """
    def __init__(self, input_dim=21, shared_dim=48, awake_hidden=64,
                 awake_hidden2=32, sleep_hidden=48, sleep_hidden2=24,
                 dropout=0.1, gate_idx=8):
        super().__init__()
        self.gate_idx = gate_idx  # sleep_markers index in feature vector

        # Shared encoder
        self.shared = nn.Linear(input_dim, shared_dim)
        self.shared_ln = nn.LayerNorm(shared_dim)

        # Awake head (larger — more complex dynamics)
        self.awake_h1 = nn.Linear(shared_dim, awake_hidden)
        self.awake_ln1 = nn.LayerNorm(awake_hidden)
        self.awake_h2 = nn.Linear(awake_hidden, awake_hidden2)
        self.awake_ln2 = nn.LayerNorm(awake_hidden2)
        self.awake_out = nn.Linear(awake_hidden2, 1)

        # Sleep head (smaller — regular exponential recovery)
        self.sleep_h1 = nn.Linear(shared_dim, sleep_hidden)
        self.sleep_ln1 = nn.LayerNorm(sleep_hidden)
        self.sleep_h2 = nn.Linear(sleep_hidden, sleep_hidden2)
        self.sleep_ln2 = nn.LayerNorm(sleep_hidden2)
        self.sleep_out = nn.Linear(sleep_hidden2, 1)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        gate = x[:, self.gate_idx].unsqueeze(-1)  # 0=awake, 1=sleep

        h = self.act(self.shared_ln(self.shared(x)))
        h = self.dropout(h)

        a = self.act(self.awake_ln1(self.awake_h1(h)))
        a = self.dropout(a)
        a = self.act(self.awake_ln2(self.awake_h2(a)))
        awake_delta = self.awake_out(a)

        s = self.act(self.sleep_ln1(self.sleep_h1(h)))
        s = self.dropout(s)
        s = self.act(self.sleep_ln2(self.sleep_h2(s)))
        sleep_delta = self.sleep_out(s)

        delta = gate * sleep_delta + (1 - gate) * awake_delta
        return delta
