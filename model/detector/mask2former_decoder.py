"""
Mask2Former Transformer Decoder
Based on official Mask2Former implementation from Facebook Research

Key Features:
- Learnable query features (supervised before decoder)
- Masked attention (constrains attention to predicted mask regions)
- Multi-scale features (3 scales: 1/32, 1/16, 1/8)
- Correct layer order: Cross-Attention → Self-Attention → FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionEmbeddingSine(nn.Module):
    """
    Sine-based positional encoding for 2D feature maps
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, H, W)
            mask: (B, H, W) optional
        Returns:
            pos: (B, C, H, W) positional encoding
        """
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        return pos


# ============================================================================
# Activation Function Helper
# ============================================================================

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ============================================================================
# Attention Layers (from official Mask2Former)
# ============================================================================

class SelfAttentionLayer(nn.Module):
    """Self-attention layer"""
    
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer (used for masked attention)"""
    
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    """Feed-forward network layer"""
    
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


# ============================================================================
# MLP (Multi-Layer Perceptron)
# ============================================================================

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ============================================================================
# Mask2Former Transformer Decoder
# ============================================================================

class Mask2FormerTransformerDecoder(nn.Module):
    """
    Mask2Former Transformer Decoder with Masked Attention
    
    This implementation follows the official Mask2Former code exactly.
    
    Key components:
    1. Learnable query features + learnable query positional embeddings
    2. Masked attention (attention constrained within predicted mask regions)
    3. Multi-scale features (round-robin across 3 scales)
    4. Layer order: Cross-Attention(Masked) → Self-Attention → FFN
    5. Predictions from learnable queries before entering decoder
    """
    
    def __init__(
        self,
        in_channels: int,
        mask_classification: bool = True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool = False,
    ):
        """
        Args:
            in_channels: channels of input features from pixel decoder
            mask_classification: whether to predict class labels (always True for Mask2Former)
            num_classes: number of classes (NOT including background)
            hidden_dim: transformer hidden dimension
            num_queries: number of object queries
            nheads: number of attention heads
            dim_feedforward: FFN hidden dimension
            dec_layers: number of decoder layers
            pre_norm: use pre-norm or post-norm
            mask_dim: mask feature dimension
            enforce_input_project: always use input projection even if dims match
        """
        super().__init__()
        
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        
        # ========== Positional Encoding ==========
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # ========== Transformer Decoder Layers ==========
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # ========== Query Embeddings (CRITICAL) ==========
        self.num_queries = num_queries
        # Learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # Learnable query positional embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # ========== Level Embeddings ==========
        # Always use 3 scales
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        
        # ========== Input Projection ==========
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Sequential())
        
        # ========== Prediction Heads ==========
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
    
    def forward(self, x, mask_features, mask=None):
        """
        Args:
            x: list of multi-scale features from pixel decoder
               [(B,C,H/32,W/32), (B,C,H/16,W/16), (B,C,H/8,W/8)]
            mask_features: (B, mask_dim, H, W) high-res features for mask prediction
            mask: not used (kept for compatibility)
            
        Returns:
            dict with:
                - pred_logits: (B, Q, num_classes+1)
                - pred_masks: (B, Q, H, W)
                - aux_outputs: list of intermediate predictions
        """
        # x is a list of multi-scale features
        assert len(x) == self.num_feature_levels
        
        src = []
        pos = []
        size_list = []
        
        # Disable mask (not used)
        del mask
        
        # Process each scale
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            # Positional encoding
            pos.append(self.pe_layer(x[i], None).flatten(2))
            # Input projection + level embedding
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            
            # Flatten NxCxHxW to HWxNxC (for transformer)
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)
        
        _, bs, _ = src[0].shape
        
        # ========== Initialize Queries ==========
        # QxNxC format
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        
        predictions_class = []
        predictions_mask = []
        
        # ========== Prediction from learnable query features (before decoder) ==========
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        # ========== Transformer Decoder Layers ==========
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            # Prevent attending to all-masked regions
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # 1. Cross-Attention (with mask from previous layer)
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed
            )
            
            # 2. Self-Attention
            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # 3. FFN
            output = self.transformer_ffn_layers[i](output)
            
            # Predict for next layer
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        
        assert len(predictions_class) == self.num_layers + 1
        
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask
            )
        }
        return out
    
    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        """
        Generate predictions and attention mask for next layer
        
        Args:
            output: (Q, B, C) query features
            mask_features: (B, mask_dim, H, W)
            attn_mask_target_size: (H, W) for attention mask
            
        Returns:
            outputs_class: (B, Q, num_classes+1)
            outputs_mask: (B, Q, H, W)
            attn_mask: (B*num_heads, Q, HW) boolean mask for attention
        """
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)  # (B, Q, C)
        
        # Class prediction
        outputs_class = self.class_embed(decoder_output)
        
        # Mask prediction via einsum
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        
        # ========== Generate Masked Attention Mask ==========
        # Resize mask prediction to attention target size
        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False
        )
        
        # Convert to boolean mask for attention
        # True = do NOT attend, False = attend
        # Attend where sigmoid(mask) > 0.5
        attn_mask = (
            attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1) < 0.5
        ).bool()
        attn_mask = attn_mask.detach()
        
        return outputs_class, outputs_mask, attn_mask
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        """
        Prepare auxiliary outputs for deep supervision
        """
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


if __name__ == "__main__":
    print("Testing Mask2Former Transformer Decoder...")
    
    # Create decoder
    decoder = Mask2FormerTransformerDecoder(
        in_channels=256,
        num_classes=2,
        hidden_dim=256,
        num_queries=30,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=6,
        pre_norm=False,
        mask_dim=256,
    )
    
    # Test forward
    bs = 2
    multi_scale_features = [
        torch.randn(bs, 256, 16, 16),  # 1/32
        torch.randn(bs, 256, 32, 32),  # 1/16
        torch.randn(bs, 256, 64, 64),  # 1/8
    ]
    mask_features = torch.randn(bs, 256, 128, 128)
    
    with torch.no_grad():
        output = decoder(multi_scale_features, mask_features)
    
    print(f"pred_logits: {output['pred_logits'].shape}")
    print(f"pred_masks: {output['pred_masks'].shape}")
    print(f"aux_outputs: {len(output['aux_outputs'])}")
    print("✓ Test passed!")
