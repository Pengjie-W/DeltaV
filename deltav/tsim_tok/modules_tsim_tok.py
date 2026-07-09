import torch
from typing import Optional
from torch import Tensor, nn
import torch.nn.functional as F
import copy, math
from einops import rearrange, repeat
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
# from .modules.attention import Attention3
from .modules.ops.modules import MSDeformAttn
from copy import deepcopy
from .modules.attention import Attention3_pos, Attention3_rope, CrossAttention
from .modules.attention import MemoryEfficientCrossAttention, XFORMERS_IS_AVAILBLE
from .modules.mlp import SwiGLUFFNFused

def build_mlp(input_dim,hidden_dim, output_dim, num_layers):

    return MLP(input_dim, hidden_dim, output_dim, num_layers)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        
        super().__init__()
        
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers - 1):
            if i < 1:
                layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        
        hidden_dim = input_dim if num_layers == 1 else hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        layers.append(nn.LayerNorm(output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, rope_1d=False):
        super().__init__()

        # self attention
        self.n_heads = n_heads
        self.rope_base = 10000
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.rope_1d=rope_1d
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm1_delta = nn.LayerNorm(d_model)
        self.norm1_delta = self.norm1
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = Attention3_rope(d_model, n_heads, dropout=dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        # self.norm3_delta = nn.LayerNorm(d_model)
        self.norm3_delta = self.norm3 

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def forward_ffn(self, src):

        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src
    @staticmethod
    def _get_ref_points_for_L(L: int, device, dtype, num_feature_levels: int):
        # L = s*s
        s = int(math.isqrt(L))
        assert s * s == L
        valid_ratios = torch.ones((1, 1, 2), device=device, dtype=dtype)  # (1,1,2)
        # Reuse get_reference_points from the Encoder.
        ref = DeformableTransformerEncoder.get_reference_points(((s, s),), valid_ratios, device)  # (1, L, 1, 2)
        # repeat to feature levels: (1, L, n_levels, 2)
        ref = ref.repeat(1, 1, num_feature_levels, 1)
        return ref  # (1, L, n_levels, 2)
    def _cross_attn_grouped_delta_bs(
            self,
            src_delta_list: List[torch.Tensor],     # list of (L_i, C)
            pos_delta_list: List[torch.Tensor],     # list of (L_i, C)
            kv_feature_delta: torch.Tensor,         # (N_delta, sum_kv, C)
            spatial_shapes_delta: torch.Tensor,     # (n_levels, 2)
            level_start_index_delta: torch.Tensor,  # (n_levels+1,)
            num_feature_levels: int,
            max_batch_size: int = 2,               # split each group into sub-batches
        ) -> List[torch.Tensor]:
        """
        Run MSDeformAttn on the delta queries grouped by L, then scatter the
        results back into a list preserving the original order.
        Within a group of equal L, process in chunks of max_batch_size to
        limit peak memory usage.
        """
        n_delta = len(src_delta_list)
        if n_delta == 0:
            return []

        device = src_delta_list[0].device
        dtype = src_delta_list[0].dtype

        lengths = torch.tensor([t.shape[0] for t in src_delta_list], device=device, dtype=torch.long)
        out_list: List[torch.Tensor] = [None] * n_delta  # type: ignore

        # Group entries that share the same L.
        for L in lengths.unique().tolist():
            L = int(L)
            idx = (lengths == L).nonzero(as_tuple=False).squeeze(1)  # (g,)
            g = idx.numel()
            if g == 0:
                continue

            # Precompute ref for this L in float32 (more stable); each chunk only expands it.
            ref_1 = self._get_ref_points_for_L(
                L,
                device=device,
                dtype=torch.float32,
                num_feature_levels=num_feature_levels
            )  # (1, L, n_levels, 2) or (L, n_levels, 2) depending on the implementation

            # Normalize to (1, L, n_levels, 2) for easier expand
            if ref_1.dim() == 3:
                ref_1 = ref_1.unsqueeze(0)

            # Split the group into chunks of max_batch_size
            for start in range(0, g, max_batch_size):
                end = min(start + max_batch_size, g)
                idx_chunk = idx[start:end]                 # (b,)
                idx_list = idx_chunk.tolist()
                b = idx_chunk.numel()

                # stack: (b, L, C)
                q = torch.stack([src_delta_list[i] for i in idx_list], dim=0).to(dtype=dtype)
                p = torch.stack([pos_delta_list[i] for i in idx_list], dim=0).to(dtype=dtype)

                # ref: (b, L, n_levels, 2)
                ref = ref_1.expand(b, -1, -1, -1).contiguous()

                # kv: (b, S_kv, C)
                kv_g = kv_feature_delta[idx_chunk]

                with torch.autocast('cuda', dtype=torch.float32, enabled=True):
                    q2 = self.cross_attn(
                        self.with_pos_embed(q, p),
                        ref,
                        kv_g,
                        spatial_shapes_delta,
                        level_start_index_delta
                    )  # (b, L, C)

                # Scatter back into the list using the original indices in idx_chunk
                for j, ii in enumerate(idx_list):
                    out_list[ii] = q2[j]

        return out_list
    def _cross_attn_grouped_delta(
        self,
        src_delta_list: List[torch.Tensor],     # list of (L_i, C)
        pos_delta_list: List[torch.Tensor],     # list of (L_i, C)
        kv_feature_delta: torch.Tensor,         # (N_delta, sum_kv, C)  (flattened by the caller)
        spatial_shapes_delta: torch.Tensor,     # (n_levels, 2)
        level_start_index_delta: torch.Tensor,  # (n_levels+1,)
        num_feature_levels: int,
    ) -> List[torch.Tensor]:
        """
        Run MSDeformAttn on the delta queries grouped by L, then scatter the
        results back into a list preserving the original order.
        """
        n_delta = len(src_delta_list)
        if n_delta == 0:
            return []

        device = src_delta_list[0].device
        dtype = src_delta_list[0].dtype

        lengths = torch.tensor([t.shape[0] for t in src_delta_list], device=device, dtype=torch.long)
        out_list = [None] * n_delta

        # Group entries that share the same L.
        for L in lengths.unique().tolist():
            L = int(L)
            idx = (lengths == L).nonzero(as_tuple=False).squeeze(1)  # (g,)
            if idx.numel() == 0:
                continue

            # stack: (g, L, C)
            q = torch.stack([src_delta_list[i] for i in idx.tolist()], dim=0)
            p = torch.stack([pos_delta_list[i] for i in idx.tolist()], dim=0)

            # ref points: (g, L, n_levels, 2)
            ref_1 = self._get_ref_points_for_L(L, device=device, dtype=torch.float32,  # ref in float32 is more stable
                                               num_feature_levels=num_feature_levels)
            ref = ref_1.expand(q.shape[0], -1, -1, -1).contiguous()

            # Index kv_feature_delta along the delta-image batch dim for this group.
            # kv_feature_delta has shape (N_delta, S_kv, C)
            kv_g = kv_feature_delta[idx]  # (g, S_kv, C)

            with torch.autocast('cuda', dtype=torch.float32, enabled=True):
                q2 = self.cross_attn(
                    self.with_pos_embed(q, p),
                    ref,
                    kv_g,
                    spatial_shapes_delta,
                    level_start_index_delta
                )  # (g, L, C)

            # Scatter back into the list
            for j, ii in enumerate(idx.tolist()):
                out_list[ii] = q2[j]

        return out_list


    def _cross_attn_delta(
        self,
        src_delta_list: List[torch.Tensor],     # list of (L_i, C)
        pos_delta_list: List[torch.Tensor],     # list of (L_i, C)
        kv_feature_delta: torch.Tensor,         # (N_delta, sum_kv, C)  (flattened by the caller)
        spatial_shapes_delta: torch.Tensor,     # (n_levels, 2)
        level_start_index_delta: torch.Tensor,  # (n_levels+1,)
        num_feature_levels: int,
    ) -> List[torch.Tensor]:
        """
        Run MSDeformAttn on the delta queries one at a time, sequentially.
        """
        n_delta = len(src_delta_list)
        if n_delta == 0:
            return []

        device = src_delta_list[0].device
        out_list = []

        # Process sequentially, one at a time
        for i in range(n_delta):
            # Take the current query and pos and add a batch dim -> (1, L_i, C)
            q = src_delta_list[i].unsqueeze(0)
            p = pos_delta_list[i].unsqueeze(0)

            L = q.shape[1]

            # Build the corresponding ref points, final shape (1, L_i, n_levels, 2)
            ref_1 = self._get_ref_points_for_L(
                L,
                device=device,
                dtype=torch.float32,  # ref in float32 is more stable
                num_feature_levels=num_feature_levels
            )
            ref = ref_1.expand(1, -1, -1, -1).contiguous()

            # Take the kv_feature for this sample and add a batch dim -> (1, S_kv, C)
            kv_i = kv_feature_delta[i].unsqueeze(0)

            # Run cross_attn
            with torch.autocast('cuda', dtype=torch.float32, enabled=True):
                q2 = self.cross_attn(
                    self.with_pos_embed(q, p),
                    ref,
                    kv_i,
                    spatial_shapes_delta,
                    level_start_index_delta
                )  # output shape (1, L_i, C)

            # Drop the batch dim -> (L_i, C) and append to the result list
            out_list.append(q2.squeeze(0))

        return out_list

    def forward(
        self,
        src,                         # (bs, T_first, C)
        src_delta_list,              # List[(L_i, C)]
        pos,                         # (bs, T_first, C)
        pos_delta_list,              # List[(L_i, C)]
        reference_points,            # (bs, T_first, n_levels, 2) precomputed
        kv_feature,                  # (bs, S_kv, C)
        kv_feature_delta,            # (N_delta, S_kv, C)
        input_spatial_shape,         # (n_levels, 2)
        input_spatial_shape_delta,   # (n_levels, 2)
        level_start_index,           # (n_levels+1,)
        level_start_index_delta,     # (n_levels+1,)
        # pos_past,
        num_images,                  # list[int] length bs
        num_feature_levels: int,
    ):
        # ====== first image cross_attn ======
        with torch.autocast('cuda', dtype=torch.float32, enabled=True):
            src2 = self.cross_attn(self.with_pos_embed(src, pos), reference_points, kv_feature,
                                   input_spatial_shape, level_start_index)
        src = self.norm1(src + self.dropout1(src2))

        # ====== delta cross_attn: grouped by L ======
        if len(src_delta_list) > 0:
            src2_delta_list = self._cross_attn_grouped_delta(
                src_delta_list, pos_delta_list,
                kv_feature_delta, input_spatial_shape_delta, level_start_index_delta,
                num_feature_levels=num_feature_levels,
            )
            # residual + norm (per delta-image)
            new_src_delta_list = []
            for s, s2 in zip(src_delta_list, src2_delta_list):
                s = self.norm1_delta(s + self.dropout1(s2))
                new_src_delta_list.append(s)
            src_delta_list = new_src_delta_list

        # ====== self_attn: concatenate the delta tokens after each sample in original order ======
        bs, T_first, C = src.shape
        device = src.device

        # rest_counts: number of delta-images per sample
        num_images_t = torch.tensor(num_images, device=device, dtype=torch.long) \
            if not torch.is_tensor(num_images) else num_images.to(device=device, dtype=torch.long)
        rest_counts = (num_images_t - 1).clamp(min=0)  # (bs,)
        rest_total = int(rest_counts.sum().item())
        assert rest_total == len(src_delta_list), f"rest_total({rest_total}) != len(src_delta_list)({len(src_delta_list)})"

        # Start offset of each sample's delta-images (counted by image, no fixed T_delta multiplier)
        img_offsets = torch.zeros((bs,), device=device, dtype=torch.long)
        if bs > 1:
            img_offsets[1:] = torch.cumsum(rest_counts, dim=0)[:-1]

        # Total length per sample: T_first + sum(L_i)
        lengths = []
        for b in range(bs):
            r = int(rest_counts[b].item())
            if r == 0:
                lengths.append(T_first)
            else:
                st = int(img_offsets[b].item())
                ed = st + r
                sumL = sum(int(src_delta_list[i].shape[0]) for i in range(st, ed))
                lengths.append(T_first + sumL)
        L_max = max(lengths)

        seq = src.new_zeros((bs, L_max, C))
        pos_seq = src.new_zeros((bs, L_max, C))
        key_padding_mask = torch.ones((bs, L_max), device=device, dtype=torch.bool)  # True=PAD

        # first image
        seq[:, :T_first] = src
        pos_seq[:, :T_first] = pos
        key_padding_mask[:, :T_first] = False

        # delta part: per sample concatenate in original order
        for b in range(bs):
            r = int(rest_counts[b].item())
            if r == 0:
                continue
            st = int(img_offsets[b].item())
            ed = st + r
            delta_cat = torch.cat([src_delta_list[i] for i in range(st, ed)], dim=0)       # (sumL, C)
            pos_cat   = torch.cat([pos_delta_list[i] for i in range(st, ed)], dim=0)       # (sumL, C)
            start = T_first
            end = T_first + delta_cat.shape[0]
            seq[b, start:end] = delta_cat
            pos_seq[b, start:end] = pos_cat
            key_padding_mask[b, start:end] = False

        # causal forbid mask
        L = seq.shape[1]
        causal_forbid = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)  # (L,L)
        attn_mask = causal_forbid[None].expand(bs, -1, -1).clone()
        attn_mask |= key_padding_mask[:, None, :]            # forbid PAD keys
        attn_mask = attn_mask[:, None, :, :]                 # (bs,1,L,L)

        # === build 2D rope ids: (t_id, l_id) per token ===
        # Convention:
        # - first image: t=0, l=0..T_first-1
        # - delta images (per sample): t=1,2,... in original order; within each image l=0..L_i-1
        t_ids = torch.zeros((bs, L_max), device=device, dtype=torch.long)
        l_ids = torch.zeros((bs, L_max), device=device, dtype=torch.long)

        # first image part
        l_ids[:, :T_first] = torch.arange(T_first, device=device, dtype=torch.long)[None, :]

        # delta part
        for b in range(bs):
            r = int(rest_counts[b].item())
            if r == 0:
                continue
            st = int(img_offsets[b].item())
            ed = st + r

            cursor = T_first
            # delta image index within this sample starts at 1
            for j, i in enumerate(range(st, ed), start=1):
                Li = int(src_delta_list[i].shape[0])
                t_ids[b, cursor:cursor+Li] = j
                l_ids[b, cursor:cursor+Li] = torch.arange(Li, device=device, dtype=torch.long)
                cursor += Li
            # padding stays 0 (it will be masked anyway)
        if self.rope_1d:
            l_ids=None
        with torch.autocast('cuda', dtype=torch.float32, enabled=True):
            tgt2 = self.self_attn(
                self.with_pos_embed(seq, pos_seq),      # (bs,L_max,C)
                attn_mask=attn_mask,
                rope_t_ids=t_ids,
                rope_l_ids=l_ids,
            )  # (bs,L_max,C)

        # write back first image
        src = self.norm3(src + self.dropout4(tgt2[:, :T_first, :]))
        src = self.forward_ffn(src)

        # split back the delta tokens (split by each delta-image's L_i, preserving order)
        tgt2_delta_list = []
        for b in range(bs):
            r = int(rest_counts[b].item())
            if r == 0:
                continue
            st = int(img_offsets[b].item())
            ed = st + r
            delta_part = tgt2[b, T_first:lengths[b], :]  # (sumL, C)
            split_sizes = [int(src_delta_list[i].shape[0]) for i in range(st, ed)]
            parts = torch.split(delta_part, split_sizes, dim=0)
            tgt2_delta_list.extend(list(parts))

        assert len(tgt2_delta_list) == len(src_delta_list)

        if len(src_delta_list) > 0:
            new_src_delta_list = []
            for s, a in zip(src_delta_list, tgt2_delta_list):
                s = self.norm3_delta(s + self.dropout4(a))
                s = self.forward_ffn(s)
                new_src_delta_list.append(s)
            src_delta_list = new_src_delta_list

        return src, src_delta_list



class DeformableTransformerEncoder(nn.Module):

    def __init__(self, dim, depth, dim_feedforward=1024, dropout=0.,
                 activation="relu", num_feature_levels = 2, n_heads=8, enc_n_points=9, rope_1d=False):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.layers = nn.ModuleList([
                        DeformableTransformerEncoderLayer(
                            dim, min(4 * dim, dim_feedforward), dropout, activation, 
                            num_feature_levels, n_heads, enc_n_points,rope_1d
                        )
                        for i in range(depth)
                    ])
        self.num_layers = depth

    def __len__(self,):

        return self.num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def forward(
        self,
        x,                       # (bs, C, H, W) first image queries-grid
        x_delta_list,            # List[(L_i, C)]
        pos,                     # (bs, T_first, C) or (bs, H*W, C)
        pos_delta_list,          # List[(L_i, C)]
        kv_features,             # list[level] of (bs, C, h, w)
        kv_features_delta,       # list[level] of (N_delta, C, h, w)
        num_images,              # list[int] length bs
    ):
        device = x.device
        bs, C, H, W = x.shape

        # first image ref points: fixed by H, W
        valid_ratios = x.new_ones((bs, 1, 2))
        reference_points = self.get_reference_points(((H, W),), valid_ratios, device)
        reference_points = repeat(reference_points, 'b n k c -> b n (f k) c', f=self.num_feature_levels)

        # kv flatten: first
        src_flatten, spatial_shapes = [], []
        for reference in kv_features:
            b, c, h, w = reference.shape
            spatial_shapes.append((h, w))
            src_flatten.append(reference.flatten(2).transpose(1, 2))
        src_flatten = torch.cat(src_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)))

        # kv flatten: delta (N_delta batch)
        src_flatten_delta, spatial_shapes_delta = [], []
        for reference in kv_features_delta:
            b, c, h, w = reference.shape
            spatial_shapes_delta.append((h, w))
            src_flatten_delta.append(reference.flatten(2).transpose(1, 2))
        src_flatten_delta = torch.cat(src_flatten_delta, 1)  # (N_delta, S_kv, C)
        spatial_shapes_delta = torch.as_tensor(spatial_shapes_delta, dtype=torch.long, device=device)
        level_start_index_delta = torch.cat((spatial_shapes_delta.new_zeros((1,)), spatial_shapes_delta.prod(1).cumsum(0)))

        # first image queries
        output = x.flatten(2).transpose(1, 2)  # (bs, H*W, C)
        output_delta_list = x_delta_list       # list[(L_i,C)]

        for layer in self.layers:
            output, output_delta_list = layer(
                output,
                output_delta_list,
                pos,
                pos_delta_list,
                reference_points,
                src_flatten,
                src_flatten_delta,
                spatial_shapes,
                spatial_shapes_delta,
                level_start_index,
                level_start_index_delta,
                num_images,
                num_feature_levels=self.num_feature_levels,
            )

        return output, output_delta_list


class Layer(nn.Module):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "xformer": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0, xformer=True):
        super().__init__()
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        
        attn_cls = self.ATTENTION_MODES[attn_mode]
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention3_rope(dim, num_head, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffnet = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_dim)

    def forward(self, x, queries=None, mask=None, pos=None, rope_t_ids=None, rope_l_ids=None):

        x = self.attn1(
            self.norm1(x),
            attn_mask=mask,
            pos=pos,
            rope_t_ids=rope_t_ids,
            rope_l_ids=rope_l_ids,
        ) + x

        x = self.ffnet(self.norm2(x)) + x
        return x


class Transformer(nn.Module):

    def __init__(self, layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout=0., xformer=False):
        super().__init__()
        self.depth = depth
        assert layer_type in ['normal',]
        layers = {'normal': Layer,}
        self.layers = nn.ModuleList([
            layers[layer_type](
                dim=dim,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                num_head=num_head,
                dropout=dropout,
                xformer=xformer
            )
            for i in range(depth)
        ])
    
    def __len__(self):

        return self.depth
    def forward(self, x, queries=None, mask=None, pos=None, rope_t_ids=None, rope_l_ids=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, queries, mask, pos, rope_t_ids=rope_t_ids, rope_l_ids=rope_l_ids)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        """
        z: Tensor, shape [L, C]   (L = total tokens across batch/seq you concatenated)
        return:
          z_q: [L, C]
          losses: (vq_loss, commit_loss, entropy_loss, codebook_usage)
          aux: (perplexity, min_encodings, min_encoding_indices)
        """
        assert z.dim() == 2 and z.size(-1) == self.e_dim, f"expect [L, {self.e_dim}], got {tuple(z.shape)}"

        z_flattened = z  # [L, C]

        if self.l2_norm:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)  # [n_e, C]
        else:
            embedding = self.embedding.weight  # [n_e, C]

        # d: [L, n_e]
        # (z - e)^2 = z^2 + e^2 - 2 z·e
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(embedding ** 2, dim=1).unsqueeze(0)
            - 2.0 * (z_flattened @ embedding.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)  # [L]
        z_q = embedding[min_encoding_indices]           # [L, C]

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        if self.training:
            vq_loss = torch.mean((z_q - z_flattened.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z_flattened) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # straight-through
        z_q = z_flattened + (z_q - z_flattened).detach()

        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices):
        """
        indices: LongTensor [L]
        return: [L, C]
        """
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        return embedding[indices]


class Decoder_Transformer(nn.Module):
    def __init__(self, layer_type, dim, n_base, depth,
                 num_head, mlp_dim, dim_head=64, dropout=0., num_register_tokens=4, image_size=256, decoder_norm=False, rope_1d=False):
        
        super().__init__()
        self.dim = dim
        scale = dim ** -0.52
        self.num_tokens = 1
        self.n_base=n_base
        self.num_register_tokens = num_register_tokens
        self.image_size=image_size
        # The original 576 is the patch count: the VFM input size is 336x336 and
        # the understanding encoder's patch_embed kernel_size=(14, 14), giving 24*24 = 576.
        # Qwen3VL's understanding encoder uses patch_embed kernel_size=(16, 16),
        # matching the decoder default.
        self.position_embedding = nn.Parameter(torch.randn(1, int(image_size/16)**2 + 1, dim) * scale)  # TODO: adjust to the training-time size
        self.rope_1d=rope_1d
        self.base_position_embedding = nn.Parameter(torch.randn(1, n_base, dim) * scale)
        self.delta_position_embedding = self.base_position_embedding

        self.mask_embedding = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.mask_dino = nn.Parameter(torch.randn(1, 1, dim) * scale)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * scale)

        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))

        self.transformer = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout, xformer=False)

        self.norm_post1 = nn.LayerNorm(dim)
        self.norm_post1_delta =self.norm_post1
        self.norm_post2 = nn.LayerNorm(dim)
        if decoder_norm:
            self.norm_post2_cc = nn.LayerNorm(dim)
        else:
            self.norm_post2_cc = self.norm_post1

        self.proj = nn.Linear(dim, 1024)

        self.initialize_weights()

    def initialize_weights(self):

        self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def interpolate_pos_encoding(self, x, w, h):
        # H_orig, W_orig are preset to match self.position_embedding.size(1)
        H_orig, W_orig = int(self.image_size/16), int(self.image_size/16)
        N = self.position_embedding.size(1) - 1
        if w == W_orig and h == H_orig:
            return self.position_embedding

        class_pos_embed = self.position_embedding[:, :1]
        patch_pos_embed = self.position_embedding[:, 1:]
        dim = x.shape[2]

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w + 0.1, h + 0.1

        m = patch_pos_embed.reshape(1, W_orig, H_orig, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, W_orig, H_orig, dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / W_orig, h0 / H_orig),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        pos_embed = torch.cat((class_pos_embed, patch_pos_embed.flatten(2).transpose(1, 2)), dim=1)
        
        return pos_embed

    def _resize_pos_embed_2d_grouped(
        self,
        num_tokens,              # list[int] for delta images, each perfect square
        mode: str = "bicubic",
        align_corners: bool = False,
    ):
        """
        Return:
        pos_list: list[Tensor], each [1, Li, C] aligned with num_tokens order
        """
        device = self.delta_position_embedding.device
        C = self.delta_position_embedding.shape[-1]

        base = self.delta_position_embedding  # [1, n_base, C]
        s0 = int(math.isqrt(self.n_base))
        assert s0 * s0 == self.n_base, f"n_base={self.n_base} must be perfect square for 2D interp"

        # [1, n_base, C] -> [1, C, s0, s0]
        base_2d = base.view(1, s0, s0, C).permute(0, 3, 1, 2).contiguous()

        Ls = torch.as_tensor(num_tokens, device=device, dtype=torch.long)
        out = [None] * len(num_tokens)

        for L in Ls.unique().tolist():
            idx = (Ls == L).nonzero(as_tuple=False).squeeze(1)
            L = int(L)
            s = int(math.isqrt(L))
            assert s * s == L, f"L={L} not perfect square"

            pe = F.interpolate(base_2d, size=(s, s), mode=mode, align_corners=align_corners)
            # [1, C, s, s] -> [1, L, C]
            pe = pe.permute(0, 2, 3, 1).contiguous().view(1, L, C)

            for j, ii in enumerate(idx.tolist()):
                out[ii] = pe  # [1, L, C]

        return out
    def _interleave_img_and_block(
        self,
        queries0,              # [n_seq, L, C] (first image tokens, already pos+LN)
        queries_delta,           # [bs_total-n_seq, delta, C] (rest image tokens, already pos+LN)
        num_images,          # list[int], len=n_seq
        block,               # [n_seq, B, C] block per sequence (e.g., cc_block or dinov2_block), already LN
        *,
        append_block_after_last=True,   # if True, also append a block after the last image
    ):
        """
        Returns:
        x_pad:   [n_seq, max_T, C]
        valid:   [n_seq, max_T] bool
        starts:  list[list[int]] each is block start index in that sequence (pre-padding coordinates)
        seq_lens:list[int]
        """
        device = queries0.device
        n_seq, L, C = queries0.shape
        assert len(num_images) == n_seq
        assert block.size(0) == n_seq and block.size(2) == C

        bs_total = int(sum(num_images))
        expected_delta = bs_total - n_seq
        if expected_delta == 0:
            assert queries_delta is None or (hasattr(queries_delta, "numel") and queries_delta.numel() == 0)
        else:
            assert queries_delta is not None
            if not isinstance(queries_delta, (list, tuple)):
                assert queries_delta.size(0) == expected_delta
            else:
                assert len(queries_delta) == expected_delta

        B = block.size(1)

        seq_tokens = []
        seq_lens = []
        starts = []

        delta_ptr = 0
        for s in range(n_seq):
            m = int(num_images[s])
            assert m >= 1

            pieces = []
            st = []

            # img0
            pieces.append(queries0[s:s+1])  # [1,L,C]

            # after img0
            st.append(sum(p.size(1) for p in pieces))
            pieces.append(block[s:s+1])   # [1,B,C]

            # imgs 1..m-1
            for k in range(1, m):
                if isinstance(queries_delta, (list, tuple)):
                    imgk = queries_delta[delta_ptr]      # already [1, Li, C]
                else:
                    imgk = queries_delta[delta_ptr:delta_ptr+1]  # [1,delta,C]
                delta_ptr += 1

                pieces.append(imgk)

                if append_block_after_last or (k < m - 1):
                    st.append(sum(p.size(1) for p in pieces))
                    pieces.append(block[s:s+1])

            toks = torch.cat(pieces, dim=1)  # [1, T_s, C]
            seq_tokens.append(toks)
            seq_lens.append(toks.size(1))
            starts.append(st)

        assert delta_ptr == expected_delta, f"queries_delta used {delta_ptr}, expected {expected_delta}"

        max_T = max(seq_lens)
        x_pad = queries0.new_zeros((n_seq, max_T, C))
        valid = torch.zeros((n_seq, max_T), dtype=torch.bool, device=device)
        for s in range(n_seq):
            T_s = seq_tokens[s].size(1)
            x_pad[s, :T_s] = seq_tokens[s][0]
            valid[s, :T_s] = True

        return x_pad, valid, starts, max_T

    def _build_causal_pad_mask(self, valid):
        """
        valid: [bs, T] bool
        returns attn_mask: [bs, T, T] bool, True means masked
        """
        bs, T = valid.shape
        device = valid.device

        causal = ~torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
        attn_mask = causal.unsqueeze(0).expand(bs, -1, -1).clone()

        pad = ~valid
        attn_mask |= pad.unsqueeze(1)  # mask padding keys
        attn_mask |= pad.unsqueeze(2)  # mask padding queries
        return attn_mask

    def _extract_block_tokens(
        self,
        x_out: torch.Tensor,   # [n_seq, T, C]
        starts,                # list[list[int]]
        *,
        rel_l: int,
        rel_r: int,
    ) -> torch.Tensor:
        """
        Extract tokens in [start+rel_l : start+rel_r] for every block occurrence.
        Return: [N, K, C], where N = total blocks (=bs_total), K = rel_r-rel_l
        """
        feats = []
        for s, st_list in enumerate(starts):
            for st in st_list:
                feats.append(x_out[s:s+1, st + rel_l: st + rel_r])  # [1,K,C]
        return torch.cat(feats, dim=0)  # [N,K,C]

    def _tokens_to_img(self, tok: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        tok: [N, H*W, C]
        return: [N, C, H, W]
        """
        return rearrange(tok, 'n (h w) c -> n c h w', h=H, w=W)
    def _build_rope_ids_segmentwise(
        self,
        *,
        n_seq: int,
        num_images: list,          # len=n_seq, each >=1
        img0_len: int,             # L (first image tokens per seq)
        delta_num_tokens: list,    # per delta-image Li in global order, len = sum(num_images)-n_seq
        block_len: int,            # cc_len or dino_len
        max_T: int,
        device,
    ):
        """
        Segment definition per sequence s:
        seg0 = img0 (len=L)
        seg1 = block (len=B)
        seg2 = img1 (len=L1)
        seg3 = block (len=B)
        seg4 = img2 (len=L2)
        seg5 = block ...
        t_id = segment index: 0,1,2,3,...
        l_id = intra-segment index: 0..seg_len-1 (reset each segment)
        """
        rope_t = torch.zeros((n_seq, max_T), device=device, dtype=torch.long)
        rope_l = torch.zeros((n_seq, max_T), device=device, dtype=torch.long)

        delta_ptr = 0
        for s in range(n_seq):
            m = int(num_images[s])
            cursor = 0
            seg = 0

            # seg0: img0
            L0 = img0_len
            rope_t[s, cursor:cursor+L0] = seg
            rope_l[s, cursor:cursor+L0] = torch.arange(L0, device=device, dtype=torch.long)
            cursor += L0
            if self.rope_1d==False:
                seg += 1

            # seg1: block after img0
            B = block_len
            rope_t[s, cursor:cursor+B] = seg
            rope_l[s, cursor:cursor+B] = torch.arange(B, device=device, dtype=torch.long)
            cursor += B
            seg += 1

            # seg2.. : (imgk, block) for k=1..m-1 (append_block_after_last=True)
            for k in range(1, m):
                Li = int(delta_num_tokens[delta_ptr])
                delta_ptr += 1

                # imgk
                rope_t[s, cursor:cursor+Li] = seg
                rope_l[s, cursor:cursor+Li] = torch.arange(Li, device=device, dtype=torch.long)
                cursor += Li
                if self.rope_1d==False:
                    seg += 1

                # block after imgk
                rope_t[s, cursor:cursor+B] = seg
                rope_l[s, cursor:cursor+B] = torch.arange(B, device=device, dtype=torch.long)
                cursor += B
                seg += 1

            # padding remains 0 (masked anyway)

        assert delta_ptr == len(delta_num_tokens), f"delta_ptr={delta_ptr}, expected={len(delta_num_tokens)}"
        return rope_t, rope_l
    def forward(self, queries, queries_delta, grid_thw, num_images, num_tokens=None):
        """
        Supports two input formats:
        - New (2D packed):
            queries:       [n_seq * n_base, C]   or [n_seq, n_base, C]
            queries_delta: [sum(num_tokens), C]  or None/empty
            num_tokens:    list[int], len = bs_total - n_seq
        - Legacy (kept for compatibility):
            queries:       [n_seq, L, C]
            queries_delta: [bs_total-n_seq, delta, C]
        """
        device = queries.device
        n_seq = len(num_images)
        bs_total = int(sum(num_images))
        expected_delta = bs_total - n_seq

        H = int(grid_thw[0][1])
        W = int(grid_thw[0][2])

        # ---------------------------------
        # 0) adapt queries: 2D packed -> [n_seq, n_base, C] (internal logic unchanged)
        # ---------------------------------
        if queries.dim() == 2:
            # [n_seq*n_base, C] -> [n_seq, n_base, C]
            assert queries.shape[0] == n_seq * self.n_base, \
                f"queries packed len {queries.shape[0]} != n_seq*n_base {n_seq*self.n_base}"
            C = queries.shape[1]
            queries = queries.view(n_seq, self.n_base, C)
        else:
            # legacy: [n_seq, L, C]
            assert queries.size(0) == n_seq
            C = queries.size(-1)

        # ---------------------------------
        # 1) image token prep (different pos + LN)
        # ---------------------------------
        # first image per seq
        L = queries.size(1)
        queries0 = self.norm_post1(queries + self.base_position_embedding)  # [n_seq, L, C]

        # remaining images (flattened across sequences)
        # New format: queries_delta is [sum Li, C] and num_tokens gives each image's Li
        if expected_delta > 0:
            assert num_tokens is not None and len(num_tokens) == expected_delta, \
                f"need num_tokens (len={expected_delta}) for delta images"

            # ---- adapt queries_delta: 2D packed -> list[[1, Li, C]], add interpolated pos + LN
            if queries_delta is None or queries_delta.numel() == 0:
                raise AssertionError("expected non-empty queries_delta for delta images")

            if queries_delta.dim() == 2:
                # split packed tokens into per-image chunks [Li, C]
                delta_chunks = list(torch.split(queries_delta, [int(x) for x in num_tokens], dim=0))
            else:
                # legacy: [expected_delta, delta, C]; this branch is essentially unused now,
                # kept only for compatibility (the current pipeline should not reach it)
                assert queries_delta.size(0) == expected_delta
                delta_chunks = [queries_delta[i] for i in range(expected_delta)]  # each [delta, C]

            # build per-image interpolated pos embeds: list[[1, Li, C]]
            pos_list = self._resize_pos_embed_2d_grouped(num_tokens, mode="bicubic", align_corners=False)

            # apply pos + LN per chunk, then stack to a list of [1, Li, C]
            queries_delta_p_list = []
            for chunk, pe in zip(delta_chunks, pos_list):
                # chunk: [Li, C] -> [1, Li, C]
                if chunk.dim() == 2:
                    chunk = chunk.unsqueeze(0)
                # add pos then LN
                q = self.norm_post1_delta(chunk + pe)  # [1, Li, C]
                queries_delta_p_list.append(q)

            # keep as list, later interleave will consume it
            queries_delta_p = queries_delta_p_list
        else:
            # no delta images
            queries_delta_p = None

        # ---------------------------------
        # 2) build cc_block per sequence (LN for cc path)
        # ---------------------------------
        z_pos = self.interpolate_pos_encoding(queries0, W, H)
        if z_pos.size(0) == 1 and n_seq > 1:
            z_pos = z_pos.expand(n_seq, -1, -1)

        cls_tok = repeat(self.cls_token, 'f ... -> (b f) ...', b=n_seq)
        mask_tok = repeat(self.mask_embedding, 'f ... -> (b f) ...', b=n_seq)
        reg_tok = repeat(self.register_tokens, 'f ... -> (b f) ...', b=n_seq)

        cc_block = torch.cat(
            (cls_tok + z_pos[:, :1], mask_tok + z_pos[:, 1:], reg_tok),
            dim=1
        )
        cc_block = self.norm_post2_cc(cc_block)
        cc_len = cc_block.size(1)
        cc_rel_l = self.num_tokens
        cc_rel_r = cc_len - self.num_register_tokens

        # ---------------------------------
        # 3) interleave tokens: [img, cc, img, cc, ...]
        #     only _interleave_img_and_block's delta handling is extended to support a list
        # ---------------------------------
        x_pad, valid, cc_starts, max_T = self._interleave_img_and_block(
            queries0=queries0,
            queries_delta=queries_delta_p,  # may now be list[[1,Li,C]]
            num_images=num_images,
            block=cc_block,
            append_block_after_last=True
        )

        attn_mask = self._build_causal_pad_mask(valid)
        attn_mask = attn_mask[:, None, :, :]
        max_T = int(max_T)

        # === build 2D rope ids for x_pad ===
        # img0_len = L = queries.size(1) = self.n_base
        img0_len = L  # length of queries0 (usually n_base)
        delta_num_tokens = [int(x) for x in num_tokens] if expected_delta > 0 else []

        rope_t_ids, rope_l_ids = self._build_rope_ids_segmentwise(
            n_seq=n_seq,
            num_images=num_images,
            img0_len=img0_len,
            delta_num_tokens=delta_num_tokens,
            block_len=cc_len,     # note: this is the cc_block length
            max_T=max_T,
            device=device,
        )
        if self.rope_1d:
            rope_l_ids = None
        x_out = self.transformer(
            x_pad,
            mask=attn_mask,
            rope_t_ids=rope_t_ids,
            rope_l_ids=rope_l_ids,
        )

        cc_tokens = self._extract_block_tokens(
            x_out, cc_starts,
            rel_l=self.num_tokens,
            rel_r=cc_len - self.num_register_tokens
        )

        z_img = self._tokens_to_img(cc_tokens, H, W)

        # ---------------------------------
        # 4) training dinov2 path (unchanged; the interleave delta input also supports a list)
        # ---------------------------------
        if self.training:
            dino_tokens = self.mask_dino + self.position_embedding[:, 1:]
            dino_tokens = repeat(dino_tokens, 'f ... -> (b f) ...', b=n_seq)

            cls_tok2 = repeat(self.cls_token, 'f ... -> (b f) ...', b=n_seq)
            reg_tok2 = repeat(self.register_tokens, 'f ... -> (b f) ...', b=n_seq)
            pos_cls = self.position_embedding[:, :1]
            if pos_cls.size(0) == 1 and n_seq > 1:
                pos_cls = pos_cls.expand(n_seq, -1, -1)

            dinov2_block = torch.cat(
                (cls_tok2 + pos_cls, dino_tokens, reg_tok2),
                dim=1
            )
            dinov2_block = self.norm_post2(dinov2_block)

            dino_len = dinov2_block.size(1)

            x2_pad, valid2, dino_starts, max_T2 = self._interleave_img_and_block(
                queries0=queries0,
                queries_delta=queries_delta_p,  # same delta list
                num_images=num_images,
                block=dinov2_block,
                append_block_after_last=True
            )
            attn_mask2 = self._build_causal_pad_mask(valid2)
            attn_mask2 = attn_mask2[:, None, :, :]
            max_T2 = int(max_T2)

            rope_t_ids2, rope_l_ids2 = self._build_rope_ids_segmentwise(
                n_seq=n_seq,
                num_images=num_images,
                img0_len=img0_len,
                delta_num_tokens=delta_num_tokens,
                block_len=dino_len,   # note: this is the dinov2_block length
                max_T=max_T2,
                device=device,
            )
            if self.rope_1d:
                rope_l_ids2 = None
            x2_out = self.transformer(
                x2_pad,
                mask=attn_mask2,
                rope_t_ids=rope_t_ids2,
                rope_l_ids=rope_l_ids2,
            )

            dino_tokens = self._extract_block_tokens(
                x2_out, dino_starts,
                rel_l=self.num_tokens,
                rel_r=dino_len - self.num_register_tokens
            )

            dinov2_out = self.proj(dino_tokens)
            return z_img, dinov2_out

        return z_img, None


class Decoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=1, norm_type="group",
                 dropout=0.0, resamp_with_conv=True, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch*ch_mult[self.num_resolutions-1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

       # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)
        
        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type='group'):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)