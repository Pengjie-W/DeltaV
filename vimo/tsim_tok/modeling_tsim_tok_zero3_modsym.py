# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ZeRO-3 + gradient-checkpointing (modsym) variant of TSIMTok. Logic-identical to the
# tokenizer embedded in vimo/modeling_vimo_zero3_gc.py; only the surrounding
# duplicated backbone code was factored out (backbone imported from vimo.backbone_vimo,
# config dataclasses shared from .modeling_tsim_tok).

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

from .modules_tsim_tok_zero3_modsym import (
    build_mlp,
    DeformableTransformerEncoder,
    VectorQuantizer,
    Decoder_Transformer,
    Decoder,
)
from vimo.backbone_vimo import Qwen3VLVisionBackbone, Qwen3VLPreTrainedModel
from .modeling_tsim_tok import (
    TSIMTokExtraCfg,
    GenCfg,
    VQCfg,
    EncCfg,
    DecCfg,
    resize_grouped_tokens,
)


class TSIMTok(Qwen3VLPreTrainedModel):
    def __init__(self, config, extra_cfg: Optional[TSIMTokExtraCfg] = None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # Qwen3VLVisionModel
        self.backbone = Qwen3VLVisionBackbone(config)
        self.extra_cfg = extra_cfg or TSIMTokExtraCfg()
        print(self.extra_cfg)
        g = self.extra_cfg.gen_cfg
        vq = self.extra_cfg.vq_cfg
        enc = self.extra_cfg.enc_cfg
        dec = self.extra_cfg.dec_cfg

        self.gen_dim = g.gen_dim
        self.gen_merger = build_mlp(config.hidden_size, self.gen_dim, self.gen_dim, g.mlp_layers)
        self.gen_norm_layer = nn.LayerNorm(self.gen_dim)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.gen_deepstack_merger_list = nn.ModuleList([
            build_mlp(config.hidden_size, self.gen_dim, self.gen_dim, g.mlp_layers)
            for _ in range(len(config.deepstack_visual_indexes))
        ])
        self.gen_deepstack_norm_layers_list = nn.ModuleList([
            nn.LayerNorm(self.gen_dim)
            for _ in range(len(config.deepstack_visual_indexes))
        ])
        self.n_base = g.n_base
        self.n_delta = g.n_delta
        scale = self.gen_dim ** -0.5
        self.base_state_queries = nn.Parameter(torch.randn(1, self.n_base, self.gen_dim) * scale)
        self.delta_state_queries = self.base_state_queries
        self.base_query_pos = nn.Parameter(torch.randn(1, self.n_base, self.gen_dim) * scale)
        self.delta_query_pos = self.base_query_pos
        self.norm_pre1 = nn.LayerNorm(self.gen_dim)
        self.norm_pre2 = self.norm_pre1

        self.vision_hidden_size = config.hidden_size
        self.en_transformer = DeformableTransformerEncoder(
            self.gen_dim,
            enc.depth,
            dim_feedforward=config.hidden_size,
            num_feature_levels=enc.num_feature_levels,
            enc_n_points=enc.enc_n_points,
            rope_1d = enc.rope_1d
        )
        self.pre_slots_quant = nn.Linear(self.gen_dim, vq.codebook_slots_embed_dim)
        self.slot_quantize = VectorQuantizer(
            vq.codebook_size,
            vq.codebook_slots_embed_dim,
            vq.commit_loss_beta,
            vq.entropy_loss_ratio,
            vq.codebook_l2_norm,
            vq.codebook_show_usage,
        )

        self.post_slots_quant = nn.Linear(vq.codebook_slots_embed_dim, self.gen_dim)
        
        self.decode_transformer = Decoder_Transformer(
            layer_type='normal',
            dim=self.gen_dim,
            n_base=self.n_base,
            depth=dec.depth,
            num_head=dec.num_head,
            mlp_dim=dec.mlp_dim,
            dim_head=dec.dim_head,
            dropout=dec.dropout,
            num_register_tokens=g.num_register_tokens,
            image_size=g.image_size,
            decoder_norm=dec.decoder_norm,
            rope_1d = enc.rope_1d,
            proj_out_dim=config.hidden_size,
        )
        self.decoder = Decoder(
            ch_mult=list(dec.ch_mult),
            z_channels=dec.z_channels,
            dropout=dec.decoder_dropout
        )

        self.gradient_checkpointing = False

    def vec2tensor(self, x, grid_thw):
        # TODO: this may be brittle; verify batch-size behavior
        if x.ndim == 2:
            x = x.unsqueeze(0)
        assert x.ndim == 3
        H = int(grid_thw[0][1])
        W = int(grid_thw[0][2])
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


    def gen_encoder(self, gen_hidden_states, process_hidden_states, grid_thw, num_images, num_tokens):
        latent = gen_hidden_states
        kv_features = []
        for i, x in enumerate(process_hidden_states):         
            kv_feature = self.gen_deepstack_merger_list[i](x)
            kv_feature = self.gen_deepstack_norm_layers_list[i](kv_feature)
            kv_feature = self.vec2tensor(kv_feature, grid_thw)
            kv_features.append(kv_feature)

        kv_feature = self.gen_merger(gen_hidden_states)
        kv_feature = self.gen_norm_layer(kv_feature)
        kv_feature = self.vec2tensor(kv_feature, grid_thw)
        kv_features.append(kv_feature)

        # --- build groups ---
        kv_features_first = []
        kv_features_rest = []
        
        for feat in kv_features:
            first_imgs = []
            rest_imgs = []
            cur_offset = 0
            for n in num_images:
                # first image
                first_imgs.append(feat[cur_offset:cur_offset + 1])
                # remaining images (if any)
                if n > 1:
                    rest_imgs.append(feat[cur_offset + 1:cur_offset + n])
                cur_offset += n
            # concatenate
            kv_features_first.append(torch.cat(first_imgs, dim=0))
            if rest_imgs:
                kv_features_rest.append(torch.cat(rest_imgs, dim=0))
            else:
                # all sequences contain only a single image
                kv_features_rest.append(feat.new_empty((0, *feat.shape[1:])))

        queries_base = self.base_state_queries.repeat(len(num_images), 1, 1)  # (n_active, n_car, C)
        queries_base = self.norm_pre1(queries_base)
        # queries_base vec2tensor: explicit grid
        h = w = int(math.sqrt(self.n_base))
        queries_base = self.vec2tensor(queries_base, [[1, h, w]])
        L_max = self.delta_state_queries.shape[1] 
        s0 = int(math.isqrt(L_max))
        assert s0 * s0 == L_max
        h0 = w0 = s0

        n_active = sum(num_images) - len(num_images)

        queries_delta = self.delta_state_queries.repeat(n_active, 1, 1)  # (n_active, L_max, dim)

        if queries_delta.shape[0] > 0:
            queries_delta = self.norm_pre2(queries_delta)
        else:
            _ = self.norm_pre2(self.base_state_queries[:1, :1, :])

        # num_tokens: list, len == n_active, each is perfect square (e.g., 121, 100, ...)
        queries_delta_list = resize_grouped_tokens(queries_delta, num_tokens, h0, w0)
        delta_query_pos = self.delta_query_pos.repeat(n_active, 1, 1)  # (n_active, L_max, dim)
        delta_query_pos_list = resize_grouped_tokens(delta_query_pos, num_tokens, h0, w0)
        # queries_delta_list[i].shape == (num_tokens[i], dim)

        first_image, interleave_image = self.en_transformer(
                    queries_base,
                    queries_delta_list,
                    self.base_query_pos,
                    delta_query_pos_list,
                    kv_features_first,
                    kv_features_rest,
                    num_images,
                )
        bs, T, Cq = first_image.shape
        first_image_flat = first_image.reshape(bs * T, Cq)  # [bs*T, Cq]
        first_image_flat = self.pre_slots_quant(first_image_flat)     # [bs, T, Cq]
        first_image_quant2, first_image_emb_loss, (_, _, first_image_q_indices) = self.slot_quantize(first_image_flat)
        if isinstance(interleave_image, (list, tuple)) and len(interleave_image) > 0:
            interleave_image_cat = torch.cat(interleave_image, dim=0)  # [sum Li, dim]
        else:
            interleave_image_cat = None
        if interleave_image_cat is not None and interleave_image_cat.numel() > 0:
            interleave_image_queries = self.pre_slots_quant(interleave_image_cat)  # [sum Li, Cq]

            interleave_image_quant2, interleave_image_emb_loss, (_, _, interleave_image_q_indices) = \
                self.slot_quantize(interleave_image_queries)
            # interleave_image_quant2: [sum Li, Cq]
            # interleave_image_q_indices: [sum Li]
        else:
            _dq_in = torch.zeros(1, self.gen_dim, device=first_image_flat.device, dtype=first_image_flat.dtype)
            _dq = self.pre_slots_quant(_dq_in)
            _ = self.slot_quantize(_dq)
            interleave_image_quant2 = first_image_quant2[:0].detach()
            interleave_image_q_indices = first_image_q_indices[:0].detach()
            interleave_image_emb_loss = None
        w1 = first_image_quant2.shape[0]  # bs*T
        w2 = interleave_image_quant2.shape[0]

        if w2 > 0:
            norm = w1 + w2
            emb_loss = tuple(
                (w1 * l1 + w2 * l2) / norm if torch.is_tensor(l1) else l1
                for l1, l2 in zip(first_image_emb_loss, interleave_image_emb_loss)
            )
        else:
            emb_loss = first_image_emb_loss

        
        return (first_image_quant2, interleave_image_quant2, latent), emb_loss, (first_image_q_indices, interleave_image_q_indices)


    def gen_decoder(self, first_image_quant2, interleave_image_quant2, grid_thw, num_images, num_tokens):
        """
        first_image_quant2: [L1, Cq]
        interleave_image_quant2: [L2, Cq]  (may have 0 rows)
        """
        first_q = self.post_slots_quant(first_image_quant2)  # [L1, C]
        if interleave_image_quant2.shape[0] > 0:
            inter_q = self.post_slots_quant(interleave_image_quant2)  # [L2, C]
        else:
            inter_q = first_q.new_empty((0, first_q.shape[-1]))        # [0, C]

        # decode_transformer accepts 2D inputs here
        z, dinov2 = self.decode_transformer(first_q, inter_q, grid_thw, num_images, num_tokens)

        dec = self.decoder(z)
        return dec, dinov2

    def gen_decode_from_indices(
        self,
        first_image_q_indices,
        interleave_image_q_indices,
        grid_thw,
        num_images,
        num_tokens=None,
    ):
        """
        first_image_q_indices: [L1]
        interleave_image_q_indices: [L2] (may be 0)
        """
        first_image_slots = self.slot_quantize.get_codebook_entry(first_image_q_indices)  # [L1, Cq]

        if interleave_image_q_indices.shape[0] > 0:
            interleave_image_slots = self.slot_quantize.get_codebook_entry(interleave_image_q_indices)  # [L2, Cq]
        else:
            interleave_image_slots = first_image_slots[:0].detach()  # [0, Cq]

        dec, dinov2 = self.gen_decoder(first_image_slots, interleave_image_slots, grid_thw, num_images, num_tokens)
        return dec, dinov2

    def encode_forward(self, hidden_states: torch.Tensor, process_hidden_states: torch.Tensor, grid_thw: torch.Tensor, num_images: torch.Tensor, num_tokens: List[int], **kwargs) -> torch.Tensor:
        gen_bs = grid_thw.shape[0]
        gen_hidden_states = hidden_states.view(gen_bs, -1, self.vision_hidden_size)
        stacked_hidden_states = torch.stack(process_hidden_states)
        gen_process_hidden_states = stacked_hidden_states.view(len(process_hidden_states), gen_bs, -1, self.vision_hidden_size)
        (first_image_quant2, interleave_image_quant2, latent), emb_loss, (first_image_q_indices, interleave_image_q_indices) = self.gen_encoder(gen_hidden_states, gen_process_hidden_states, grid_thw, num_images, num_tokens)

        return first_image_q_indices, interleave_image_q_indices

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, num_images: torch.Tensor, num_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states, process_hidden_states = self.backbone(hidden_states, grid_thw, **kwargs)
        gen_bs = grid_thw.shape[0]
        gen_hidden_states = hidden_states.view(gen_bs, -1, self.vision_hidden_size)
        stacked_hidden_states = torch.stack(process_hidden_states)
        gen_process_hidden_states = stacked_hidden_states.view(len(process_hidden_states), gen_bs, -1, self.vision_hidden_size)
        (first_image_quant2, interleave_image_quant2, latent), emb_loss, (first_image_q_indices, interleave_image_q_indices) = \
            self.gen_encoder(gen_hidden_states, gen_process_hidden_states, grid_thw, num_images, num_tokens)

        if self.training:
            dec, dinov2 = self.gen_decoder(first_image_quant2, interleave_image_quant2, grid_thw, num_images, num_tokens)
        else:
            dec, dinov2 = self.gen_decode_from_indices(
                first_image_q_indices,
                interleave_image_q_indices,
                grid_thw,
                num_images,
                num_tokens=num_tokens,
            )

        return (dec, latent, dinov2), emb_loss, (first_image_q_indices, interleave_image_q_indices)


__all__ = ["TSIMTok"]
