import torch, math, pdb
from torch import Tensor
from copy import deepcopy
from .mlp import build_mlp
from torch import nn, einsum
from inspect import isfunction
from typing import Optional, Any
from einops import rearrange, repeat
from torch.nn import functional as F
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d    


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale
        
        sim = q @ k.transpose(-2, -1)
        sim = sim.softmax(dim=-1)

        out = sim @ v
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        proj = self.to_out(out)    

        return proj


class MaskedCrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):

        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask = None):
        
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        mask = repeat(mask, 'b f -> (b h) f', h = h)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale
        
        sim = torch.einsum('n m d, n k d -> n m k', q, k)

        decimal = torch.exp(sim) * (mask.unsqueeze(1)).float()
        attn = decimal / decimal.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = attn @ v
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)    

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), 
            nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask = None):

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)    



class Attention2(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float=0,
        downsample_rate = 1,
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.scale = 1 / math.sqrt(self.internal_dim / num_heads)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:

        n, b, c = x.shape
        x = rearrange(x, 'n b (h c) -> n h b c', h = self.num_heads)
        
        return x  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:

        x = rearrange(x, 'b h k c -> b k (h c)')
        return x

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None) -> Tensor:
        
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        bs, _, num, c_per_head = q.shape
        
        # Attention
        logits = q @ k.transpose(2, 3)  # B x N_heads x N_tokens x N_tokens
        logits = logits * self.scale

        attn = torch.softmax(logits,dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn


class Attention3(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float=0,
        downsample_rate = 1,
    ) -> None:

        super().__init__()

        self.dropout = dropout
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.scale = 1 / math.sqrt(self.internal_dim / num_heads)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:

        n, b, c = x.shape
        x = rearrange(x, 'n b (h c) -> n h b c', h = self.num_heads)
        
        return x  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:

        x = rearrange(x, 'b h k c -> b k (h c)')
        return x

    def forward(self, src, slots: Tensor = None, attn_mask: Tensor = None) -> Tensor:
        
        #* Input projections
        q = self.q_proj(src)
        k = self.k_proj(src)
        v = self.v_proj(src)

        #* Separate into heads
        q, k, v = map(lambda x: self._separate_heads(x, self.num_heads), (q, k, v))
        
        if attn_mask is not None:
            attn_mask = ~attn_mask

        #* Attention
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            is_causal=True if attn_mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.dropout if self.training else 0, scale=self.scale)

        #* Get output
        out = self._recombine_heads(output)
        out = self.out_proj(out) 

        return out

class Attention3_pos(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float=0,
        downsample_rate = 1,
    ) -> None:

        super().__init__()

        self.dropout = dropout
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.scale = 1 / math.sqrt(self.internal_dim / num_heads)
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:

        n, b, c = x.shape
        x = rearrange(x, 'n b (h c) -> n h b c', h = self.num_heads)
        
        return x  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:

        x = rearrange(x, 'b h k c -> b k (h c)')
        return x
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward(self, src, slots: Tensor = None, attn_mask: Tensor = None, pos: Tensor = None) -> Tensor:

        q = k = self.with_pos_embed(src, pos)
        #* Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)

        v = self.v_proj(src)

        #* Separate into heads
        q, k, v = map(lambda x: self._separate_heads(x, self.num_heads), (q, k, v))
        
        if attn_mask is not None:
            attn_mask = ~attn_mask

        #* Attention
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, 
            is_causal=True if attn_mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.dropout if self.training else 0, scale=self.scale)

        #* Get output
        out = self._recombine_heads(output)
        out = self.out_proj(out) 

        return out

def build_2d_rope_cis_from_ids(
    head_dim: int,
    t_ids: torch.Tensor,   # (B, L)
    l_ids: torch.Tensor,   # (B, L)
    theta: float = 10000.0,
    dtype=torch.float32,
):
    """
    returns: freqs_cis (B, L, head_dim//2) complex
    head_dim must be divisible by 4.
    """
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
    device = t_ids.device
    t_ids = t_ids.to(dtype=dtype)
    l_ids = l_ids.to(dtype=dtype)

    # (head_dim/4,)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4, device=device, dtype=dtype) / head_dim))

    # (B, L, head_dim/4)
    t_ang = t_ids[..., None] * freqs[None, None, :]
    l_ang = l_ids[..., None] * freqs[None, None, :]

    t_cis = torch.polar(torch.ones_like(t_ang), t_ang)  # complex
    l_cis = torch.polar(torch.ones_like(l_ang), l_ang)  # complex

    # interleave -> (B, L, head_dim/2) complex: [t0,l0,t1,l1,...]
    freqs_cis = torch.stack([t_cis, l_cis], dim=-1).reshape(t_ids.shape[0], t_ids.shape[1], -1)
    return freqs_cis

def build_1d_rope_cis_from_ids(
    head_dim: int,
    t_ids: torch.Tensor,   # (B, L)
    theta: float = 10000.0,
    dtype=torch.float32,
):
    """
    returns: freqs_cis (B, L, head_dim//2) complex
    head_dim must be divisible by 2.
    """
    assert head_dim % 2 == 0, f"head_dim must be divisible by 2, got {head_dim}"
    device = t_ids.device
    t_ids = t_ids.to(dtype=dtype)

    # 1D RoPE only needs head_dim/2 frequencies
    # (head_dim/2,)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))

    # (B, L, head_dim/2)
    ang = t_ids[..., None] * freqs[None, None, :]

    # Convert directly to complex representation; no interleaving needed
    freqs_cis = torch.polar(torch.ones_like(ang), ang)  # complex

    return freqs_cis

def apply_rotary_emb_bhlc(q, k, freqs_cis):
    """
    q,k: (B, H, L, head_dim) real
    freqs_cis: (B, L, head_dim//2) complex
    """
    B, H, L, D = q.shape
    assert D % 2 == 0
    
    orig_dtype = q.dtype

    # Cast to float32 if not already
    if orig_dtype != torch.float32:
        q = q.float()
        k = k.float()

    # (B,H,L,D/2,2) -> complex
    q_ = torch.view_as_complex(q.reshape(B, H, L, D // 2, 2).contiguous())
    k_ = torch.view_as_complex(k.reshape(B, H, L, D // 2, 2).contiguous())

    # (B,1,L,D/2) complex broadcast to heads
    freqs_cis = freqs_cis[:, None, :, :]

    q_out = q_ * freqs_cis
    k_out = k_ * freqs_cis

    # back to real: (B,H,L,D)
    q_out = torch.view_as_real(q_out).reshape(B, H, L, D)
    k_out = torch.view_as_real(k_out).reshape(B, H, L, D)

    # Cast back to the original dtype
    if orig_dtype != torch.float32:
        q_out = q_out.to(orig_dtype)
        k_out = k_out.to(orig_dtype)

    return q_out, k_out
class Attention3_rope(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0, downsample_rate=1, rope_theta=10000.0):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        assert self.internal_dim % num_heads == 0
        self.head_dim = self.internal_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def with_pos_embed(self, x, pos):
        return x if pos is None else x + pos

    def _split_heads(self, x):  # x: (B,L,internal_dim)
        B, L, C = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,Dh)

    def _merge_heads(self, x):  # x: (B,H,L,Dh)
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)  # (B,L,internal_dim)

    def forward(self, src, attn_mask=None, pos=None, rope_t_ids=None, rope_l_ids=None):
        """
        src: (B,L,C)
        pos: (B,L,C)  -> additive content bias
        rope_t_ids, rope_l_ids: (B,L) long/int
        attn_mask: (B,1,L,L) bool (True=forbid). torch SDPA expects a bool mask where True means keep,
                   so attn_mask is inverted once below.
        """
        x = self.with_pos_embed(src, pos)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(src)

        q = self._split_heads(q)  # (B,H,L,Dh)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # === 2D RoPE here ===
        if rope_t_ids is not None and rope_l_ids is not None:
            freqs_cis = build_2d_rope_cis_from_ids(
                head_dim=self.head_dim,
                t_ids=rope_t_ids,
                l_ids=rope_l_ids,
                theta=self.rope_theta,
                dtype=torch.float32,
            )
            q, k = apply_rotary_emb_bhlc(q, k, freqs_cis)
        elif rope_t_ids is not None:
            freqs_cis = build_1d_rope_cis_from_ids(
                head_dim=self.head_dim,
                t_ids=rope_t_ids,
                theta=self.rope_theta,
                dtype=torch.float32,
            )
            q, k = apply_rotary_emb_bhlc(q, k, freqs_cis)
        if attn_mask is not None:
            attn_mask = ~attn_mask  # invert mask to match SDPA convention

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=True if attn_mask is None else False,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out

import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,                 # (N, Lq, C)
        reference_points,      # IGNORE
        input_flatten,         # (N, Lk, C)  used as K/V
        input_spatial_shapes,  # IGNORE
        input_level_start_index,  # IGNORE
        input_padding_mask=None   # (N, Lk) True=pad
    ):
        # PyTorch MHA key_padding_mask: True means that key position is masked out
        out, _ = self.mha(
            query=query,
            key=input_flatten,
            value=input_flatten,
            key_padding_mask=input_padding_mask
        )
        return self.dropout(out)

class CrossAttentionAddPosK(nn.Module):
    """
    Standard cross-attention: Q comes from query, K/V come from input_flatten.
    Additionally supports adding a shared pos_k: (1, S_kv, C) to K, broadcast over the batch.
    To minimize changes, forward keeps the MSDeformAttn argument signature; reference_points/
    spatial_shapes/start_index are ignored.
    """
    def __init__(self, d_model=256, n_heads=8, dropout=0.1, add_pos_to_v: bool = False):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.add_pos_to_v = add_pos_to_v

    def forward(
        self,
        query,                 # (N, Lq, C)
        reference_points,      # ignored
        input_flatten,         # (N, S_kv, C)
        input_spatial_shapes,  # ignored
        input_level_start_index,  # ignored
        input_padding_mask=None,  # (N, S_kv) True=pad
        pos_k=None             # NEW: (1, S_kv, C) or (N, S_kv, C)
    ):
        k = input_flatten
        v = input_flatten

        if pos_k is not None:
            # Allow passing (1,S,C) or (N,S,C)
            if pos_k.dim() != 3:
                raise ValueError(f"pos_k must be 3D (B,S,C), got {pos_k.shape}")
            if pos_k.shape[0] == 1 and query.shape[0] > 1:
                pos_k = pos_k.expand(query.shape[0], -1, -1).contiguous()
            # Add pos to every K
            k = k + pos_k
            if self.add_pos_to_v:
                v = v + pos_k

        out, _ = self.mha(
            query=query,
            key=k,
            value=v,
            key_padding_mask=input_padding_mask
        )
        return self.dropout(out)