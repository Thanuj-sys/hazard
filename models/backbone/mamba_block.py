import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """
    Implements the core Mamba selective scan mechanism.
    """
    def __init__(self, d_model: int = 256, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)
        self.dt_rank = int(d_model / 16) # Rank for dt

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.activation = nn.SiLU()

        # x_proj takes in x and outputs delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # SSM parameters A
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) - where B is batch, L is sequence length, D is d_model
        Returns:
            (B, L, D)
        """
        B, L, D = x.shape
        # (B, L, D) -> (B, L, d_inner * 2)
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Conv1d expects (B, C, L)
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2) # (B, L, d_inner)
        
        x_act = self.activation(x_conv)
        
        # Projections
        x_dbl = self.x_proj(x_act) # (B, L, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # (B, L, d_inner)
        
        # Selective scan
        y = self.selective_scan_fn(x_act, dt, B_ssm, C_ssm)
        
        y = y * self.activation(z)
        out = self.out_proj(y)
        return out

    def selective_scan_fn(self, u: torch.Tensor, dt: torch.Tensor, B_ssm: torch.Tensor, C_ssm: torch.Tensor) -> torch.Tensor:
        """
        Selective scan mechanism recurrence.
        Args:
            u: (B, L, d_inner)
            dt: (B, L, d_inner)
            B_ssm: (B, L, d_state)
            C_ssm: (B, L, d_state)
        Returns:
            (B, L, d_inner)
        """
        batch, seq_len, d_inner = u.shape
        d_state = self.d_state
        
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Discretize
        dt = F.softplus(dt) # (B, L, d_inner)
        
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            A_bar = torch.exp(dt_t * A) # (B, d_inner, d_state)
            B_bar = dt_t * B_ssm[:, t, :].unsqueeze(1) # (B, d_inner, d_state)
            
            u_t = u[:, t, :].unsqueeze(-1) # (B, d_inner, 1)
            
            h = A_bar * h + B_bar * u_t
            
            y_t = torch.sum(h * C_ssm[:, t, :].unsqueeze(1), dim=-1) # (B, d_inner)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # (B, L, d_inner)
        y = y + u * self.D
        return y


class MambaBlock(nn.Module):
    """
    Wraps SelectiveSSM with residual connection, layer norm, and optional dropout
    """
    def __init__(self, d_model: int = 256, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        res = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return res + x
