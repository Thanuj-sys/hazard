import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Tuple

class GeoAttributeGrounder(nn.Module):
    """Extracts and formats geo-attributes into conditioning tokens.
    
    Ensures the language model is grounded in real terrain data,
    not hallucinating environmental claims.
    """
    def __init__(self, num_attributes: int = 5, embed_dim: int = 256):
        super().__init__()
        self.num_attributes = num_attributes
        self.embed_dim = embed_dim
        
        # Projects raw scalar geo-attributes to embedding space
        self.attribute_proj = nn.Linear(1, embed_dim)
        
        # Learnable attribute type embeddings to differentiate elevation, slope, etc.
        self.attribute_embeddings = nn.Parameter(torch.randn(1, num_attributes, embed_dim))
        
        # Expected attributes: elevation, slope, ndvi, lithology_class, aspect
        self.attribute_names = ["elevation", "slope", "ndvi", "lithology_class", "aspect"]

    def forward(self, geo_attributes: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            geo_attributes: Dictionary containing scalar tensors for each attribute.
                            Each tensor should have shape (B, 1).
                            
        Returns:
            grounding_tokens: (B, num_attributes, embed_dim)
        """
        batch_size = next(iter(geo_attributes.values())).size(0)
        device = next(iter(geo_attributes.values())).device
        
        # Collect attribute values in expected order
        attr_vals = []
        for name in self.attribute_names:
            # Fallback to zero if attribute is missing
            val = geo_attributes.get(name, torch.zeros(batch_size, 1, device=device))
            # Ensure shape is (B, 1)
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            attr_vals.append(val)
            
        # Stack to (B, num_attributes, 1)
        stacked_attrs = torch.stack(attr_vals, dim=1)
        
        # Project to (B, num_attributes, embed_dim)
        projected = self.attribute_proj(stacked_attrs)
        
        # Add attribute type embeddings
        grounding_tokens = projected + self.attribute_embeddings
        
        return grounding_tokens


class ReportStructureEncoder(nn.Module):
    """Encodes the expected report structure as learnable section tokens.
    
    Report sections (following IMD/NDMA conventions):
    1. hazard_type: Classification of observed hazard
    2. severity_assessment: Severity level with evidence
    3. terrain_analysis: DEM/slope/NDVI analysis of affected area  
    4. damage_description: Observable damage from imagery
    5. restoration_priority: Prioritized intervention recommendations
    """
    def __init__(self, num_sections: int = 5, embed_dim: int = 256):
        super().__init__()
        self.num_sections = num_sections
        self.embed_dim = embed_dim
        
        # Learnable section embeddings
        self.section_embeddings = nn.Parameter(torch.randn(1, num_sections, embed_dim))
        
        # Section-aware positional encoding (simplified as learned positional embeddings here)
        self.section_pos_enc = nn.Parameter(torch.randn(1, num_sections, embed_dim))

    def forward(self) -> torch.Tensor:
        """
        Returns:
            section_tokens: (1, num_sections, embed_dim)
        """
        return self.section_embeddings + self.section_pos_enc


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x


class LanguageReasonerHead(nn.Module):
    """Geo-grounded language generation head for hazard reports.
    
    ARCHITECTURAL CONSTRAINT: This head NEVER outputs coordinates or
    radius numbers. It generates structured natural language reports
    conditioned on extracted geo-attributes.
    
    Output is structured into sections evaluable against:
    - DAP (Damage Assessment Precision)
    - DDR (Damage Detail Recall)
    - FC (Factual Correctness)
    - RN (Recovery Necessity)
    - SC (Strategic Completeness)
    - APP (Action Priority Precision)
    """
    def __init__(self, embed_dim: int = 256, vocab_size: int = 32000, 
                 num_layers: int = 4, num_heads: int = 8, 
                 max_report_length: int = 512, num_geo_attributes: int = 5,
                 num_report_sections: int = 5, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_report_length = max_report_length
        self.num_report_sections = num_report_sections
        
        self.grounder = GeoAttributeGrounder(num_attributes=num_geo_attributes, embed_dim=embed_dim)
        self.structure_encoder = ReportStructureEncoder(num_sections=num_report_sections, embed_dim=embed_dim)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_report_length)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, 
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        self.section_names = [
            "hazard_type", "severity_assessment", "terrain_analysis", 
            "damage_description", "restoration_priority"
        ]

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, fused_sequence: torch.Tensor, geo_attributes: Dict[str, torch.Tensor], 
                target_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            fused_sequence: (B, L, embed_dim) from fusion layer
            geo_attributes: dict of terrain tensors for grounding
            target_tokens: (B, S) target token ids for teacher forcing (training)
            
        Returns:
            logits: (B, S, vocab_size) token prediction logits, or None if target_tokens is None
            report_sections: dict mapping section_name -> (B, section_len, embed_dim)
        """
        batch_size = fused_sequence.size(0)
        device = fused_sequence.device
        
        # 1. Prepare conditioning context
        # Extract grounding tokens: (B, num_attributes, embed_dim)
        grounding_tokens = self.grounder(geo_attributes)
        
        # Extract section tokens: (1, num_sections, embed_dim) -> (B, num_sections, embed_dim)
        section_tokens = self.structure_encoder().expand(batch_size, -1, -1)
        
        # Combine memory context: fused vision sequence + geo grounding + section structure
        # Total memory length = L + num_attributes + num_sections
        memory = torch.cat([fused_sequence, grounding_tokens, section_tokens], dim=1)
        
        # 2. Extract section features for evaluation (mock up fixed sized section representation)
        report_sections = {}
        for i, name in enumerate(self.section_names):
            # For simplicity, representing each section with its corresponding section token from memory
            # In a full implementation, you might map the generated sequence parts back to sections
            # Shape: (B, 1, embed_dim)
            report_sections[name] = section_tokens[:, i:i+1, :]
            
        # 3. Autoregressive language generation (if targets provided)
        logits = None
        if target_tokens is not None:
            seq_len = target_tokens.size(1)
            # Token embedding and positional encoding
            # (B, S, embed_dim)
            tgt = self.token_embedding(target_tokens)
            
            # Positional encoding expects (S, B, E) if batch_first=False, but our PositionalEncoding
            # uses that format. We'll transpose for pos encoding then back.
            tgt = tgt.transpose(0, 1)
            tgt = self.pos_encoder(tgt)
            tgt = tgt.transpose(0, 1)
            
            # Causal mask for autoregressive generation
            tgt_mask = self._generate_square_subsequent_mask(seq_len, device)
            
            # Decode: (B, S, embed_dim)
            output = self.transformer_decoder(
                tgt=tgt, memory=memory, tgt_mask=tgt_mask
            )
            
            # Project to vocabulary: (B, S, vocab_size)
            logits = self.output_proj(output)
            
        return logits, report_sections

    @torch.no_grad()
    def generate(self, fused_sequence: torch.Tensor, geo_attributes: Dict[str, torch.Tensor], 
                 max_length: int = 512, temperature: float = 0.7, top_k: int = 50, 
                 start_token_id: int = 1, eos_token_id: int = 2) -> torch.Tensor:
        """
        Autoregressive generation for inference.
        
        Args:
            fused_sequence: (B, L, embed_dim)
            geo_attributes: dict of terrain tensors
            max_length: max tokens to generate
            temperature: softmax temperature
            top_k: top-k sampling threshold
            start_token_id: ID for <BOS> token
            eos_token_id: ID for <EOS> token
            
        Returns:
            generated_token_ids: (B, generated_length)
        """
        batch_size = fused_sequence.size(0)
        device = fused_sequence.device
        
        # Prepare memory
        grounding_tokens = self.grounder(geo_attributes)
        section_tokens = self.structure_encoder().expand(batch_size, -1, -1)
        memory = torch.cat([fused_sequence, grounding_tokens, section_tokens], dim=1)
        
        # Initialize generation
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Embed current sequence
            tgt = self.token_embedding(generated)
            tgt = tgt.transpose(0, 1)
            tgt = self.pos_encoder(tgt)
            tgt = tgt.transpose(0, 1)
            
            # Causal mask
            seq_len = generated.size(1)
            tgt_mask = self._generate_square_subsequent_mask(seq_len, device)
            
            # Decode
            output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            
            # Get logits for the last token
            next_token_logits = self.output_proj(output[:, -1, :]) / temperature
            
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            is_finished |= (next_token.squeeze(-1) == eos_token_id)
            if is_finished.all():
                break
                
        return generated
