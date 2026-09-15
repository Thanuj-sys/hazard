import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class H3CellEmbedding(nn.Module):
    """
    Converts H3 cell indices into dense embeddings for different resolutions.
    """
    def __init__(self, num_resolutions: int = 3, max_cells_per_res: int = 10000, embed_dim: int = 256):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.max_cells_per_res = max_cells_per_res
        self.embed_dim = embed_dim
        
        # Create an embedding layer for each resolution
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=max_cells_per_res, embedding_dim=embed_dim)
            for _ in range(num_resolutions)
        ])
        
    def forward(self, h3_indices_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            h3_indices_list: list of length `num_resolutions`, where each element is a
                             tensor of H3 cell indices of shape (B, num_cells_at_res).
        
        Returns:
            list of length `num_resolutions`, where each element is a tensor of 
            embeddings of shape (B, num_cells_at_res, embed_dim).
        """
        out = []
        for i, indices in enumerate(h3_indices_list):
            # indices shape: (B, num_cells_at_res)
            # emb shape: (B, num_cells_at_res, embed_dim)
            emb = self.embeddings[i](indices)
            out.append(emb)
        return out

class TerrainContextEncoder(nn.Module):
    """
    Encodes static geo-attributes (DEM, slope, NDVI, lithology, aspect) into a dense vector.
    """
    def __init__(self, num_geo_attributes: int = 5, embed_dim: int = 256):
        super().__init__()
        # 4 continuous attributes: elevation, slope, ndvi, aspect
        self.continuous_proj = nn.Linear(4, embed_dim // 2)
        
        # 1 categorical attribute: lithology_class (assuming 100 classes max for embedding)
        self.lithology_emb = nn.Embedding(100, embed_dim // 2)
        
        # Final projection to combine continuous and categorical
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, geo_attributes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            geo_attributes: dict with keys 'elevation', 'slope', 'ndvi', 'lithology_class', 'aspect'.
                            Each tensor should be of shape (B, 1) or (B,).
                            
        Returns:
            Tensor of shape (B, embed_dim)
        """
        # Ensure correct shapes (B, 1)
        elevation = geo_attributes['elevation'].view(-1, 1)
        slope = geo_attributes['slope'].view(-1, 1)
        ndvi = geo_attributes['ndvi'].view(-1, 1)
        aspect = geo_attributes['aspect'].view(-1, 1)
        lithology = geo_attributes['lithology_class'].view(-1).long()
        
        # (B, 4) -> (B, embed_dim // 2)
        cont_features = torch.cat([elevation, slope, ndvi, aspect], dim=-1)
        cont_emb = self.continuous_proj(cont_features)
        
        # (B, embed_dim // 2)
        lith_emb = self.lithology_emb(lithology)
        
        # (B, embed_dim)
        combined = torch.cat([cont_emb, lith_emb], dim=-1)
        out = self.final_proj(combined)
        return out

class GeospatialPromptEncoder(nn.Module):
    """
    Main encoder that produces fusable geospatial tokens.
    
    Resolution choices:
    - Resolution 4 (~22km edge): macro-region context, captures inter-district hazard correlation
    - Resolution 6 (~3.2km edge): district-level, matches 0.1-30km hazard radius range
    - Resolution 8 (~0.46km edge): fine-grained epicenter localization
    """
    def __init__(self, embed_dim: int = 256, num_resolutions: int = 3, max_cells_per_res: int = 10000, 
                 num_geo_attributes: int = 5, max_prompt_tokens: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_resolutions = num_resolutions
        self.max_prompt_tokens = max_prompt_tokens
        
        # 1. H3CellEmbedding
        self.h3_embedding = H3CellEmbedding(num_resolutions, max_cells_per_res, embed_dim)
        
        # 2. TerrainContextEncoder
        self.terrain_encoder = TerrainContextEncoder(num_geo_attributes, embed_dim)
        
        # 3. Learnable resolution-level positional embeddings
        self.res_pos_embeddings = nn.Parameter(torch.randn(num_resolutions, embed_dim))
        
        # 4. Cross-resolution aggregation via self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # 5. Final linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def get_output_dim(self) -> int:
        return self.embed_dim
        
    def forward(self, h3_indices: Dict[int, torch.Tensor], geo_attributes: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h3_indices: dict mapping resolution index (0 to num_resolutions-1) to cell index tensors, 
                        shape per res: (B, num_cells_at_res)
            geo_attributes: dict with terrain attribute tensors
            
        Returns:
            geo_tokens: (B, max_prompt_tokens, embed_dim)
            geo_mask: (B, max_prompt_tokens) boolean mask
        """
        B = list(h3_indices.values())[0].size(0)
        
        # Extract and sort by resolution key
        sorted_res_keys = sorted(list(h3_indices.keys()))
        indices_list = [h3_indices[k] for k in sorted_res_keys]
        
        # (B, num_cells_at_res, embed_dim) for each res
        h3_embs = self.h3_embedding(indices_list)
        
        # Add resolution positional embeddings
        all_tokens = []
        for i, emb in enumerate(h3_embs):
            # emb: (B, num_cells_at_res, embed_dim)
            # self.res_pos_embeddings[i]: (embed_dim,)
            emb = emb + self.res_pos_embeddings[i].view(1, 1, -1)
            all_tokens.append(emb)
            
        # Terrain token
        # (B, embed_dim)
        terrain_emb = self.terrain_encoder(geo_attributes)
        # (B, 1, embed_dim)
        terrain_emb = terrain_emb.unsqueeze(1)
        all_tokens.append(terrain_emb)
        
        # Concatenate tokens
        # Shape: (B, total_tokens, embed_dim)
        concat_tokens = torch.cat(all_tokens, dim=1)
        total_tokens = concat_tokens.size(1)
        
        # Cross-resolution attention
        attn_out, _ = self.self_attn(concat_tokens, concat_tokens, concat_tokens)
        
        # Final projection
        # Shape: (B, total_tokens, embed_dim)
        proj_out = self.out_proj(attn_out)
        
        # Pad or truncate to max_prompt_tokens
        device = proj_out.device
        geo_tokens = torch.zeros(B, self.max_prompt_tokens, self.embed_dim, device=device)
        geo_mask = torch.zeros(B, self.max_prompt_tokens, dtype=torch.bool, device=device)
        
        if total_tokens > self.max_prompt_tokens:
            geo_tokens = proj_out[:, :self.max_prompt_tokens, :]
            geo_mask[:, :] = True
        else:
            geo_tokens[:, :total_tokens, :] = proj_out
            geo_mask[:, :total_tokens] = True
            
        return geo_tokens, geo_mask
