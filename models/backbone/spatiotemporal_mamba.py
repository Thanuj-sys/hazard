import torch
import torch.nn as nn
from .mamba_block import MambaBlock

class PatchEmbedding(nn.Module):
    """
    Converts multi-band multi-temporal satellite imagery into patch tokens.
    """
    def __init__(self, num_channels: int = 10, patch_size: int = 16, embed_dim: int = 256, img_size: int = 512, max_seq_len: int = 30):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        self.num_patches_per_dim = img_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Embeddings
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, 1, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, T * num_patches, embed_dim)
        """
        B, T, C, H, W = x.shape
        # Flatten time and batch to process through Conv2d
        x = x.reshape(B * T, C, H, W) # (B*T, C, H, W)
        
        x = self.proj(x) # (B*T, embed_dim, H/P, W/P)
        
        x = x.flatten(2).transpose(1, 2) # (B*T, num_patches, embed_dim)
        x = x.reshape(B, T, -1, self.embed_dim) # (B, T, num_patches, embed_dim)
        
        # Add embeddings
        x = x + self.temporal_pos_embed[:, :T, :, :]
        x = x + self.spatial_pos_embed
        
        # Flatten sequence
        x = x.reshape(B, -1, self.embed_dim) # (B, T * num_patches, embed_dim)
        return x

class TemporalMaskHandler(nn.Module):
    """
    Handles missing frames in the 30-day window using quality flags.
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.missing_frame_token = nn.Parameter(torch.zeros(embed_dim))

    def create_temporal_mask(self, quality_flags: torch.Tensor) -> torch.Tensor:
        """
        Creates boolean mask of valid timesteps.
        Args:
            quality_flags: (B, T) 
        Returns:
            (B, T) boolean mask where True indicates valid frame.
        """
        return quality_flags.bool()

    def forward(self, x: torch.Tensor, temporal_mask: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Args:
            x: (B, T * num_patches, embed_dim)
            temporal_mask: (B, T) boolean mask, True is valid
            num_patches: int
        Returns:
            x: updated tokens
        """
        B, seq_len, D = x.shape
        T = seq_len // num_patches
        x = x.reshape(B, T, num_patches, D)
        
        # Create a mask for broadcasting: (B, T, 1, 1)
        mask = temporal_mask.view(B, T, 1, 1).expand(-1, -1, num_patches, D)
        
        # Replace invalid tokens with missing_frame_token
        missing_tokens = self.missing_frame_token.view(1, 1, 1, D).expand(B, T, num_patches, D)
        x = torch.where(mask, x, missing_tokens)
        
        x = x.reshape(B, seq_len, D)
        return x

class SpatiotemporalMambaBackbone(nn.Module):
    """
    Full backbone combining patch embedding, temporal masking, and stacked Mamba blocks.
    """
    def __init__(self, num_channels=10, patch_size=16, embed_dim=256, num_layers=6, d_state=16, d_conv=4, expand=2, max_seq_len=30, img_size=512, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.img_size = img_size
        
        self.patch_embed = PatchEmbedding(
            num_channels=num_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            img_size=img_size,
            max_seq_len=max_seq_len
        )
        
        self.temporal_mask_handler = TemporalMaskHandler(embed_dim=embed_dim)
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

    def get_output_dim(self) -> int:
        return self.embed_dim

    def forward(self, x: torch.Tensor, quality_flags: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
            quality_flags: (B, T)
        Returns:
            visual_tokens: (B, seq_len, embed_dim)
            temporal_mask: (B, T)
        """
        # 1. Patch Embedding
        # (B, T*num_patches, embed_dim)
        x_embed = self.patch_embed(x)
        
        # 2. Temporal Masking
        temporal_mask = self.temporal_mask_handler.create_temporal_mask(quality_flags)
        num_patches = self.patch_embed.num_patches
        x_embed = self.temporal_mask_handler(x_embed, temporal_mask, num_patches)
        
        # 3. Stacked Mamba Blocks
        for layer in self.layers:
            x_embed = layer(x_embed)
            
        # 4. Final LayerNorm
        visual_tokens = self.norm(x_embed)
        
        return visual_tokens, temporal_mask
