# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.

import torch
import torch.nn.functional as F

from einops import rearrange
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.models.attention import BasicTransformerBlock
from .dit_attention import BasicTransformerBlock as _BasicTransformerBlock   # attention blocks from animatediff
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UpBlock2D

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class MutualSelfAttentionControl(AttentionBase):

    def __init__(self, total_steps=50, hijack_init_state=True, with_negative_guidance=False, appearance_control_alpha=0.5, mode='enqueue'):
        """
        Mutual self-attention control for Stable-Diffusion MODEl
        Args:
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.hijack = hijack_init_state
        self.with_negative_guidance = with_negative_guidance
        
        # alpha: mutual self attention intensity
        # TODO: make alpha learnable
        self.alpha = appearance_control_alpha
        self.GLOBAL_ATTN_QUEUE = []
        assert mode in ['enqueue', 'dequeue']
        MODE = mode
    
    def attn_batch(self, q, k, v, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def mutual_self_attn(self, q, k, v, num_heads, **kwargs):
        q_tgt, q_src = q.chunk(2)
        k_tgt, k_src = k.chunk(2)
        v_tgt, v_src = v.chunk(2)
        
        # out_tgt = self.attn_batch(q_tgt, k_src, v_src, num_heads, **kwargs) * self.alpha + \
        #           self.attn_batch(q_tgt, k_tgt, v_tgt, num_heads, **kwargs) * (1 - self.alpha)
        out_tgt = self.attn_batch(q_tgt, torch.cat([k_tgt, k_src], dim=1), torch.cat([v_tgt, v_src], dim=1), num_heads, **kwargs)
        out_src = self.attn_batch(q_src, k_src, v_src, num_heads, **kwargs)
        out = torch.cat([out_tgt, out_src], dim=0)
        return out
    
    def mutual_self_attn_wq(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if self.MODE == 'dequeue' and len(self.kv_queue) > 0:
            k_src, v_src = self.kv_queue.pop(0)
            out = self.attn_batch(q, torch.cat([k, k_src], dim=1), torch.cat([v, v_src], dim=1), num_heads, **kwargs)
            return out
        else:
            self.kv_queue.append([k.clone(), v.clone()])
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
    
    def get_queue(self):
        return self.GLOBAL_ATTN_QUEUE
    
    def set_queue(self, attn_queue):
        self.GLOBAL_ATTN_QUEUE = attn_queue
    
    def clear_queue(self):
        self.GLOBAL_ATTN_QUEUE = []
    
    def to(self, dtype):
        self.GLOBAL_ATTN_QUEUE = [p.to(dtype) for p in self.GLOBAL_ATTN_QUEUE]

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


class ReferenceAttentionControl():
    
    # 12 4 12 is down mid up
    _cross_unet_layer_num = 12
    def __init__(self, 
                 unet,
                 mode="write",
                 do_classifier_free_guidance=False,
                 attention_auto_machine_weight = float('inf'),
                 gn_auto_machine_weight = 1.0,
                 style_fidelity = 1.0,
                 reference_attn=True,
                 reference_adain=False,
                 fusion_blocks="midup",
                 batch_size=1,
                 clip_length=8,
                 is_image=False,
                 ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode, 
            do_classifier_free_guidance,
            clip_length,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks=fusion_blocks,
            batch_size=batch_size,
            is_image=is_image,
        )

    def register_reference_hooks(
            self, 
            mode, 
            do_classifier_free_guidance,
            clip_length,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            dtype=torch.float16,
            batch_size=1, 
            num_images_per_prompt=1, 
            device=torch.device("cpu"), 
            fusion_blocks='midup',
            is_image=False,
        ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype=dtype
        if do_classifier_free_guidance:
            # uc_mask = (
            #     torch.Tensor([1] * batch_size * num_images_per_prompt * 16 + [0] * batch_size * num_images_per_prompt * 16)
            #     .to(device)
            #     .bool()
            # )

            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * clip_length + [0] * batch_size * num_images_per_prompt * clip_length)
                    .to(device)
                    .bool()
            )
            

        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )
        
        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

             # 0. Retrieve lora scale.
            lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            if MODE == "write":
                self.bank.append(norm_hidden_states.clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                if self.use_ada_layer_norm_zero:
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                hidden_states = attn_output + hidden_states

                # 2.5 GLIGEN Control
                if gligen_kwargs is not None:
                    hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
                # 2.5 ends

                # 3. Cross-Attention
                if self.attn2 is not None:
                    # Cross-Attention
                    norm_hidden_states = (
                        self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                    )

                    hidden_states = (
                        self.attn2(
                            norm_hidden_states, 
                            encoder_hidden_states=encoder_hidden_states, 
                            attention_mask=encoder_attention_mask
                            **cross_attention_kwargs,
                        )
                        + hidden_states
                    )

                # Feed-forward
                norm_hidden_states = self.norm3(hidden_states)

                if self.use_ada_layer_norm_zero:
                    # if video_length is not None:
                    #     scale_mlp = rearrange(scale_mlp.unsqueeze(1).repeat(1, video_length, 1), 'b f c -> (b f) c')
                    #     shift_mlp = rearrange(shift_mlp.unsqueeze(1).repeat(1, video_length, 1), 'b f c -> (b f) c')
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                        raise ValueError(
                            f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                        )

                    num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
                    ff_output = torch.cat(
                        [
                            self.ff(hid_slice, scale=lora_scale)
                            for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                        ],
                        dim=self._chunk_dim,
                    )
                else:
                    ff_output = self.ff(norm_hidden_states, scale=lora_scale)

                if self.use_ada_layer_norm_zero:
                    # if video_length is not None:
                    #     gate_mlp = rearrange(gate_mlp.unsqueeze(1).repeat(1, video_length, 1), 'b f c -> (b f) c')
                    ff_output = gate_mlp.unsqueeze(1) * ff_output

                hidden_states = ff_output + hidden_states
                return hidden_states
            if MODE == "read":
                if not is_image:
                    self.bank = [rearrange(d.unsqueeze(1).repeat(1, video_length, 1, 1), "b t l c -> (b t) l c")[:hidden_states.shape[0]] for d in self.bank]
                # LLZ: animate anyone
                # modify_norm_hidden_states = torch.cat(
                #     [norm_hidden_states] + self.bank, dim=1)
                # hidden_states_uc = self.attn1(modify_norm_hidden_states,
                #                               encoder_hidden_states=modify_norm_hidden_states,
                #                               attention_mask=attention_mask)[:, :hidden_states.shape[-2], :] + hidden_states
                # print("torch.cat([norm_hidden_states] + self.bank, dim=1) shape is", \
                #     torch.cat([norm_hidden_states] + self.bank, dim=1).shape)
                # print("norm_hidden_states shape is", norm_hidden_states.shape)
                # 2. Prepare GLIGEN inputs
                if self.unet_use_cross_frame_attention:
                    attn_output_uc = self.attn1(
                        norm_hidden_states, 
                        encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                        # encoder_hidden_states=None,
                        attention_mask=attention_mask, 
                        video_length=video_length)
                else:
                    attn_output_uc = self.attn1(
                        norm_hidden_states, 
                        encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                        # encoder_hidden_states=None,
                        attention_mask=attention_mask)

                if self.use_ada_layer_norm_zero:
                    attn_output_uc = gate_msa.unsqueeze(1) * attn_output_uc
                hidden_states_uc = attn_output_uc + hidden_states

                # hidden_states_uc = self.attn1(norm_hidden_states, 
                #                             encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                #                             attention_mask=attention_mask) + hidden_states
                
                hidden_states_c = hidden_states_uc.clone()
                _uc_mask = uc_mask.clone()
                if do_classifier_free_guidance:
                    if hidden_states.shape[0] != _uc_mask.shape[0]:
                        _uc_mask = (
                            torch.Tensor([1] * (hidden_states.shape[0]//2) + [0] * (hidden_states.shape[0]//2))
                            .to(device)
                            .bool()
                        )
                    attn_output_uc_r = self.attn1(
                        norm_hidden_states[_uc_mask],
                        encoder_hidden_states=norm_hidden_states[_uc_mask],
                        attention_mask=attention_mask,
                    )
                    if self.use_ada_layer_norm_zero:
                        attn_output_uc_r = gate_msa[_uc_mask].unsqueeze(1) * attn_output_uc_r
                    hidden_states_c[_uc_mask] = attn_output_uc_r + hidden_states[_uc_mask]

                    # hidden_states_c[_uc_mask] = self.attn1(
                    #     norm_hidden_states[_uc_mask],
                    #     encoder_hidden_states=norm_hidden_states[_uc_mask],
                    #     attention_mask=attention_mask,
                    # ) + hidden_states[_uc_mask]

                hidden_states = hidden_states_c.clone()
                # 2.5 GLIGEN Control
                if gligen_kwargs is not None:
                    hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
                # 2.5 ends
                                                        
                self.bank.clear()
                if self.attn2 is not None:
                    # Cross-Attention
                    norm_hidden_states = (
                        self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                    )
                    hidden_states = (
                        self.attn2(
                            norm_hidden_states, 
                            encoder_hidden_states=encoder_hidden_states, 
                            attention_mask=attention_mask,
                            **cross_attention_kwargs,
                        )
                        + hidden_states
                    )

                # Feed-forward
                norm_hidden_states = self.norm3(hidden_states)

                if self.use_ada_layer_norm_zero:
                    # if video_length is not None:
                    #     scale_mlp = rearrange(scale_mlp.unsqueeze(1).repeat(1, video_length, 1), 'b f c -> (b f) c')
                    #     shift_mlp = rearrange(shift_mlp.unsqueeze(1).repeat(1, video_length, 1), 'b f c -> (b f) c')
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                        raise ValueError(
                            f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                        )

                    num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
                    ff_output = torch.cat(
                        [
                            self.ff(hid_slice, scale=lora_scale)
                            for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
                        ],
                        dim=self._chunk_dim,
                    )
                else:
                    ff_output = self.ff(norm_hidden_states, scale=lora_scale)

                if self.use_ada_layer_norm_zero:
                    # if video_length is not None:
                    #     gate_mlp = rearrange(gate_mlp.unsqueeze(1).repeat(1, video_length, 1), 'b f c -> (b f) c')
                    ff_output = gate_mlp.unsqueeze(1) * ff_output

                hidden_states = ff_output + hidden_states

                # Temporal-Attention
                if self.unet_use_temporal_attention and not is_image:
                    d = hidden_states.shape[1]
                    hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
                    norm_hidden_states = (
                        self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
                    )
                    hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
                    hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

                return hidden_states


        if self.reference_attn:
            l = self._cross_unet_layer_num
            if self.fusion_blocks == "midup":
                attn_modules = self.unet.transformer_blocks[l:]
            elif self.fusion_blocks == "full":
                attn_modules = self.unet.transformer_blocks            
            # attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            # print("attn_modules is", len(attn_modules))
            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))
    
    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            l = self._cross_unet_layer_num
            if self.fusion_blocks == "midup":
                reader_attn_modules = self.unet.transformer_blocks[l:]
                writer_attn_modules = writer.unet.transformer_blocks[l:]
            elif self.fusion_blocks == "full":
                reader_attn_modules = self.unet.transformer_blocks
                writer_attn_modules = writer.unet.transformer_blocks
            # reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])    
            # writer_attn_modules = sorted(writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            # print("len writer_attn_modules is", len(writer_attn_modules), "len reader_attn_modules", len(reader_attn_modules))
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]
                # for x in r.bank:
                #     print('x shape is', x.shape)
                # w.bank.clear()
        if self.reference_adain:
            reader_gn_modules = [self.unet.mid_block]
            
            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                reader_gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                reader_gn_modules.append(module)
                
            writer_gn_modules = [writer.unet.mid_block]
            
            down_blocks = writer.unet.down_blocks
            for w, module in enumerate(down_blocks):
                writer_gn_modules.append(module)

            up_blocks = writer.unet.up_blocks
            for w, module in enumerate(up_blocks):
                writer_gn_modules.append(module)
            
            for r, w in zip(reader_gn_modules, writer_gn_modules):
                if len(w.mean_bank) > 0 and isinstance(w.mean_bank[0], list):
                    r.mean_bank = [[v.clone().to(dtype) for v in vl] for vl in w.mean_bank]
                    r.var_bank = [[v.clone().to(dtype) for v in vl] for vl in w.var_bank]
                else:
                    r.mean_bank = [v.clone().to(dtype) for v in w.mean_bank]
                    r.var_bank = [v.clone().to(dtype) for v in w.var_bank]
    
    def clear(self):
        if self.reference_attn:
            l = self._cross_unet_layer_num
            if self.fusion_blocks == "midup":
                reader_attn_modules = self.unet.transformer_blocks[l:]
            elif self.fusion_blocks == "full":
                reader_attn_modules = self.unet.transformer_blocks
            # reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            for r in reader_attn_modules:
                r.bank.clear()
        if self.reference_adain:
            reader_gn_modules = [self.unet.mid_block]
            
            down_blocks = self.unet.down_blocks
            for w, module in enumerate(down_blocks):
                reader_gn_modules.append(module)

            up_blocks = self.unet.up_blocks
            for w, module in enumerate(up_blocks):
                reader_gn_modules.append(module)
            
            for r in reader_gn_modules:
                r.mean_bank.clear()
                r.var_bank.clear()
            