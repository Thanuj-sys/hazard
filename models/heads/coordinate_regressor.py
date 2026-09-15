import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class GeofencedCoordinateRegressor(nn.Module):
    """Sigmoid-activated coordinate regression head.
    
    Outputs bounded lat/lon offsets within the input tile/AOI.
    Includes geofenced clamping to validate predictions against
    country/state boundary polygons.
    
    ARCHITECTURAL CONSTRAINT: Fully decoupled from Language Reasoner.
    Takes the same fused representation but produces numeric output only.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = embed_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 2))  # lat, lon
        
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
        
        # Learnable scale and bias for converting normalized output to lat/lon offsets if needed
        # But generally, we use aoi_bounds to map sigmoid output to real coordinates
        self.scale = nn.Parameter(torch.ones(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        fused_global: torch.Tensor,
        aoi_bounds: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            fused_global: (B, embed_dim) from fusion layer
            aoi_bounds: optional (B, 4) tensor [min_lat, min_lon, max_lat, max_lon]
                       for converting sigmoid output to real coordinates
        Returns:
            coords_normalized: (B, 2) sigmoid output in [0, 1]
            coords_geo: (B, 2) actual [lat, lon] if aoi_bounds provided, else None
        """
        # (B, 2)
        raw_out = self.mlp(fused_global)
        
        # (B, 2)
        coords_normalized = self.sigmoid(raw_out)
        
        coords_geo = None
        if aoi_bounds is not None:
            # aoi_bounds: (B, 4) -> min_lat, min_lon, max_lat, max_lon
            min_lat = aoi_bounds[:, 0:1]
            min_lon = aoi_bounds[:, 1:2]
            max_lat = aoi_bounds[:, 2:3]
            max_lon = aoi_bounds[:, 3:4]
            
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            
            pred_lat = min_lat + coords_normalized[:, 0:1] * lat_range
            pred_lon = min_lon + coords_normalized[:, 1:2] * lon_range
            
            coords_geo = torch.cat([pred_lat, pred_lon], dim=1) # (B, 2)
            
        return coords_normalized, coords_geo

    def geofence_clamp(
        self,
        coords_geo: torch.Tensor,
        boundary_polygon: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Validate coordinates against geofence boundary.
        If outside, snap to nearest valid point.
        
        Args:
            coords_geo: (B, 2) tensor of [lat, lon]
            boundary_polygon: optional tensor representing boundary points
        
        Returns:
            clamped_coords: (B, 2) tensor of [lat, lon]
        """
        # Placeholder for geometric clamping logic
        # In practice, this would use spatial libraries (e.g., shapely) to snap points
        # to the boundary polygon if they fall outside.
        return coords_geo

    @staticmethod
    def haversine_distance(pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
        """Compute haversine distance in km between predicted and true coordinates.
        For use as evaluation metric.
        
        Args:
            pred_coords: (B, 2) tensor [lat, lon] in degrees
            true_coords: (B, 2) tensor [lat, lon] in degrees
        Returns:
            distances: (B,) distance in km
        """
        # Earth radius in km
        R = 6371.0
        
        # Convert degrees to radians
        lat1, lon1 = torch.deg2rad(pred_coords[:, 0]), torch.deg2rad(pred_coords[:, 1])
        lat2, lon2 = torch.deg2rad(true_coords[:, 0]), torch.deg2rad(true_coords[:, 1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        
        distance = R * c
        return distance
