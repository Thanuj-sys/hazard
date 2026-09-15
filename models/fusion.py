import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: queries attend to key-value pairs."""
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # LayerNorm on query and kv inputs (pre-norm)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        
        # Multi-head cross attention
        # PyTorch MultiheadAttention handles q, k, v projections internally
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN LayerNorm
        self.norm_ffn = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, kv: torch.Tensor, kv_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query: (B, Lq, D)
            kv: (B, Lkv, D)
            kv_mask: (B, Lkv) boolean mask where True means valid token.
        Returns:
            (B, Lq, D)
        """
        # Pre-norm
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(kv)
        
        # In MultiheadAttention, key_padding_mask has True for padded/ignored elements.
        # If kv_mask has True = valid token, we need to invert it.
        key_padding_mask = None
        if kv_mask is not None:
            key_padding_mask = ~kv_mask # True where invalid
            
        attn_out, _ = self.cross_attn(
            query=q_norm, 
            key=kv_norm, 
            value=kv_norm, 
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection
        out = query + self.dropout(attn_out)
        
        # FFN
        ffn_out = self.ffn(self.norm_ffn(out))
        
        # Residual connection
        out = out + ffn_out
        
        return out


class CrossModalityFusionLayer(nn.Module):
    """Fuses visual backbone tokens with geospatial prompt encoder tokens.
    
    CRITICAL: Output is head-agnostic. Do NOT bake in assumptions favoring
    any one downstream head. The three heads (Language Reasoner, Coordinate
    Regressor, Radius Estimator) are DECOUPLED after this point.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.vis_to_geo_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.geo_to_vis_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Final projection layer to ensure consistent output dim
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, visual_tokens: torch.Tensor, geo_tokens: torch.Tensor, geo_mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_tokens: (B, L_vis, embed_dim) from Mamba backbone
            geo_tokens: (B, L_geo, embed_dim) from GeospatialPromptEncoder  
            geo_mask: (B, L_geo) boolean, True = valid token
            
        Returns:
            fused_sequence: (B, L_vis + L_geo, embed_dim) - full fused token sequence
                           (for Language Reasoner which needs full sequence)
            fused_global: (B, embed_dim) - global pooled representation
                         (for Coordinate Regressor and Radius Estimator)
        """
        # Bidirectional cross-attention
        curr_vis = visual_tokens
        curr_geo = geo_tokens
        
        # Apply layers sequentially
        for v2g, g2v in zip(self.vis_to_geo_layers, self.geo_to_vis_layers):
            # vis queries geo
            new_vis = v2g(curr_vis, curr_geo, kv_mask=geo_mask)
            
            # geo queries vis
            # Note: visual tokens typically don't have a mask in this setup (always valid)
            new_geo = g2v(curr_geo, curr_vis)
            
            curr_vis = new_vis
            curr_geo = new_geo
            
        # Concatenate full sequence
        # (B, L_vis + L_geo, embed_dim)
        fused_sequence = torch.cat([curr_vis, curr_geo], dim=1)
        fused_sequence = self.final_proj(fused_sequence)
        
        # Global pooled representation
        # Pool visual tokens (could also pool entire sequence, but vis tokens have global context)
        fused_global = curr_vis.mean(dim=1)  # (B, embed_dim)
        
        return fused_sequence, fused_global

    def get_output_dim(self) -> int:
        return self.embed_dim
