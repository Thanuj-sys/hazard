import torch
import torch.nn as nn
import yaml
from typing import Dict, Any, Optional, Tuple, List

# Import sub-modules using correct paths
from .backbone.spatiotemporal_mamba import SpatiotemporalMambaBackbone
from .prompt_encoder import GeospatialPromptEncoder
from .fusion import CrossModalityFusionLayer
from .heads.language_reasoner import LanguageReasonerHead
from .heads.coordinate_regressor import GeofencedCoordinateRegressor
from .heads.radius_estimator import FluidRadiusDynamicsEstimator, HazardClassificationHead


class MultiHazardVLM(nn.Module):
    """MultiHazard-VLM: End-to-end vision-language pipeline for multi-hazard mapping.

    Architecture has THREE DECOUPLED heads after a shared fusion layer:
    1. Language Reasoner (reports & advice) - grounded in geo-attributes, NEVER outputs coords/radius
    2. Coordinate Regressor (sigmoid head) - outputs bounded lat/lon offsets
    3. Radius Estimator (MLP + ReLU) - outputs impact radius in km, clamped

    Plus a HazardClassificationHead for hazard type prediction.
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.config = self._load_config(config_path)

        model_cfg = self.config.get('model', self.config)
        embed_dim = model_cfg.get('embed_dim', 256)
        num_heads = model_cfg.get('num_heads', 8)
        num_input_channels = model_cfg.get('num_input_channels', 10)
        num_geo_attributes = model_cfg.get('num_geo_attributes', 5)
        patch_size = model_cfg.get('patch_size', 16)
        max_seq_len = model_cfg.get('max_seq_len', 30)
        img_size = model_cfg.get('img_size', 512)

        # 1. Visual Backbone — Spatiotemporal Mamba
        self.backbone = SpatiotemporalMambaBackbone(
            num_channels=num_input_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=6,
            max_seq_len=max_seq_len,
            img_size=img_size,
        )

        # 2. Geospatial Prompt Encoder — H3 hierarchical grid
        self.prompt_encoder = GeospatialPromptEncoder(
            embed_dim=embed_dim,
            num_geo_attributes=num_geo_attributes,
        )

        # 3. Cross-Attention Fusion Layer
        self.fusion = CrossModalityFusionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        # 4. Decoupled Heads (parallel, independent after fusion)
        self.classification_head = HazardClassificationHead(
            embed_dim=embed_dim,
            num_classes=3,  # flood, wildfire, landslide
        )
        self.coordinate_regressor = GeofencedCoordinateRegressor(
            embed_dim=embed_dim,
        )
        self.radius_estimator = FluidRadiusDynamicsEstimator(
            embed_dim=embed_dim,
        )
        self.language_reasoner = LanguageReasonerHead(
            embed_dim=embed_dim,
            num_geo_attributes=num_geo_attributes,
        )

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or return defaults."""
        if config_path is None:
            return {
                'model': {
                    'embed_dim': 256,
                    'num_heads': 8,
                    'num_input_channels': 10,
                    'num_geo_attributes': 5,
                    'patch_size': 16,
                    'max_seq_len': 30,
                }
            }
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def forward(
        self,
        images: torch.Tensor,
        quality_flags: torch.Tensor,
        h3_indices: Dict[int, torch.Tensor],
        geo_attributes: Dict[str, torch.Tensor],
        target_tokens: Optional[torch.Tensor] = None,
        aoi_bounds: Optional[torch.Tensor] = None,
        hazard_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass of the MultiHazardVLM.

        Args:
            images: (B, T, C, H, W) spatiotemporal satellite imagery
            quality_flags: (B, T) boolean mask, True = valid frame
            h3_indices: dict mapping resolution -> cell index tensors per batch
            geo_attributes: dict with keys 'elevation', 'slope', 'ndvi', 'aspect', 'lithology_class'
            target_tokens: (B, S) optional target token ids for teacher forcing
            aoi_bounds: (B, 4) optional [min_lat, min_lon, max_lat, max_lon]
            hazard_type: (B,) optional int tensor for per-hazard radius clamping

        Returns:
            Dict containing all head outputs.
        """
        # 1. Visual feature extraction
        # visual_tokens: (B, T*num_patches, embed_dim), temporal_mask: (B, T)
        visual_tokens, temporal_mask = self.backbone(images, quality_flags)

        # 2. Geospatial prompt encoding
        # geo_tokens: (B, max_prompt_tokens, embed_dim), geo_mask: (B, max_prompt_tokens)
        geo_tokens, geo_mask = self.prompt_encoder(h3_indices, geo_attributes)

        # 3. Cross-modality fusion
        # fused_sequence: (B, L_vis + L_geo, embed_dim), fused_global: (B, embed_dim)
        fused_sequence, fused_global = self.fusion(
            visual_tokens, geo_tokens, geo_mask
        )

        # 4. Decoupled Heads (parallel, independent)
        # Classification: returns (logits, probabilities)
        hazard_logits, hazard_probs = self.classification_head(fused_global)

        # Coordinate Regression: returns (coords_normalized, coords_geo)
        coords_normalized, coords_geo = self.coordinate_regressor(
            fused_global, aoi_bounds
        )

        # Radius Estimation: returns radius_km (B, 1)
        radius_km = self.radius_estimator(fused_global, hazard_type)

        # Language Reasoning: returns (logits, report_sections)
        language_logits, report_sections = self.language_reasoner(
            fused_sequence, geo_attributes, target_tokens
        )

        return {
            'hazard_logits': hazard_logits,
            'hazard_probs': hazard_probs,
            'coords_normalized': coords_normalized,
            'coords_geo': coords_geo,
            'radius_km': radius_km,
            'language_logits': language_logits,
            'report_sections': report_sections,
            'temporal_mask': temporal_mask,
        }

    @torch.no_grad()
    def generate_report(
        self,
        images: torch.Tensor,
        quality_flags: torch.Tensor,
        h3_indices: Dict[int, torch.Tensor],
        geo_attributes: Dict[str, torch.Tensor],
        aoi_bounds: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Generate full hazard report for inference.

        Returns:
            Dict with hazard_class, epicenter_coordinates, impact_radius_km, report tokens.
        """
        self.eval()

        visual_tokens, temporal_mask = self.backbone(images, quality_flags)
        geo_tokens, geo_mask = self.prompt_encoder(h3_indices, geo_attributes)
        fused_sequence, fused_global = self.fusion(
            visual_tokens, geo_tokens, geo_mask
        )

        # Heads
        hazard_logits, hazard_probs = self.classification_head(fused_global)
        hazard_class = torch.argmax(hazard_logits, dim=-1)

        coords_normalized, coords_geo = self.coordinate_regressor(
            fused_global, aoi_bounds
        )
        radius_km = self.radius_estimator(fused_global, hazard_class)

        # Autoregressive generation
        generated_tokens = self.language_reasoner.generate(
            fused_sequence, geo_attributes
        )

        return {
            'hazard_class': hazard_class,
            'hazard_probs': hazard_probs,
            'epicenter_coords_normalized': coords_normalized,
            'epicenter_coords_geo': coords_geo,
            'impact_radius_km': radius_km,
            'report_tokens': generated_tokens,
        }
