import torch
import torch.nn as nn
from typing import Tuple, Optional


class FluidRadiusDynamicsEstimator(nn.Module):
    """MLP + ReLU radius regression head with per-hazard clamping.
    
    Outputs physical impact radius in km.
    Clamped to [0.5, 150] km (updated range).
    
    NOTE (open design question flagged for team): One fixed radius range
    across floods, wildfire, and landslides may not be physically
    appropriate. Different hazards have very different spatial scales:
    - Landslide runout: typically 0.1-5 km
    - Wildfire spread: typically 1-50 km  
    - Flood extent: typically 5-150 km
    Per-hazard-type sub-ranges are implemented but configurable.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        radius_min: float = 0.5,
        radius_max: float = 150.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.radius_min = radius_min
        self.radius_max = radius_max
        
        # Hazard-specific bounds: 0=flood, 1=wildfire, 2=landslide
        self.hazard_bounds = {
            0: (1.0, 150.0),
            1: (0.5, 80.0),
            2: (0.1, 10.0)
        }
        
        layers = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        fused_global: torch.Tensor,
        hazard_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            fused_global: (B, embed_dim) from fusion layer
            hazard_type: optional (B,) int tensor (0=flood, 1=wildfire, 2=landslide)
                        for per-hazard clamping
        Returns:
            radius_km: (B, 1) predicted impact radius in km
        """
        # (B, 1)
        raw_radius = self.mlp(fused_global)
        
        # Use ReLU to ensure non-negativity
        radius_km = torch.relu(raw_radius)
        
        # Clamp output
        if hazard_type is not None:
            # Per-hazard clamping
            clamped = radius_km.clone()
            for b in range(hazard_type.size(0)):
                h_type = int(hazard_type[b].item())
                h_min, h_max = self.hazard_bounds.get(h_type, (self.radius_min, self.radius_max))
                clamped[b, 0] = torch.clamp(radius_km[b, 0], min=h_min, max=h_max)
            radius_km = clamped
        else:
            # Global clamping
            radius_km = torch.clamp(radius_km, min=self.radius_min, max=self.radius_max)
            
        return radius_km


class HazardClassificationHead(nn.Module):
    """Softmax classification head for hazard type prediction.
    
    P(Hazard) = Softmax(W_c @ Z_fuse)
    Classes: flood (0), wildfire (1), landslide (2)
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused_global: (B, embed_dim) from fusion layer
        Returns:
            logits: (B, num_classes)
            probabilities: (B, num_classes)
        """
        logits = self.mlp(fused_global)
        probabilities = self.softmax(logits)
        
        return logits, probabilities
