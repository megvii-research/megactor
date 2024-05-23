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

from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward

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
    _refer_cross_layer_num = 12
    _control_cross_layer_num = 12
    def __init__(self, 
                 unet,
                 model_type="",
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
        assert model_type in ["unet", "appearencenet", "controlnet"]
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.model_type = model_type
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode, 
            model_type,
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
        model_type,
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
        MODEL_TYPE = model_type
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
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ):
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            if MODE == "write":
                self.bank.append(norm_hidden_states.clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                if self.norm_type == "ada_norm_zero":
                    attn_output = gate_msa.unsqueeze(1) * attn_output
                elif self.norm_type == "ada_norm_single":
                    attn_output = gate_msa * attn_output

                hidden_states = attn_output + hidden_states
                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)

                # 1.2 GLIGEN Control
                if gligen_kwargs is not None:
                    hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

                # 3. Cross-Attention
                if self.attn2 is not None:
                    if self.norm_type == "ada_norm":
                        norm_hidden_states = self.norm2(hidden_states, timestep)
                    elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                        norm_hidden_states = self.norm2(hidden_states)
                    elif self.norm_type == "ada_norm_single":
                        # For PixArt norm2 isn't applied here:
                        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                        norm_hidden_states = hidden_states
                    elif self.norm_type == "ada_norm_continuous":
                        norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                    else:
                        raise ValueError("Incorrect norm")

                    if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                        norm_hidden_states = self.pos_embed(norm_hidden_states)

                    # print("norm_hidden_states shape is", norm_hidden_states.shape)
                    # print("encoder_hidden_states shape is", encoder_hidden_states.shape)
                    # print("encoder_attention_mask shape is", encoder_attention_mask.shape)
                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    hidden_states = attn_output + hidden_states

                # 4. Feed-forward
                # i2vgen doesn't have this norm 🤷‍♂️
                if self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif not self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm3(hidden_states)

                if self.norm_type == "ada_norm_zero":
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                if self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm2(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states)

                if self.norm_type == "ada_norm_zero":
                    ff_output = gate_mlp.unsqueeze(1) * ff_output
                elif self.norm_type == "ada_norm_single":
                    ff_output = gate_mlp * ff_output

                hidden_states = ff_output + hidden_states
                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)

                return hidden_states
            if MODE == "read":
                # print("before self.bank shape", self.bank[0].shape)
                # print("before hidden_states shape", hidden_states.shape)
                self.bank = [rearrange(d.unsqueeze(1).repeat(1, hidden_states.shape[0] // d.shape[0], 1, 1), "b t l c -> (b t) l c")[:hidden_states.shape[0]] for d in self.bank]
                # print("after self.bank shape", self.bank[0].shape)
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
                attn_output_uc = self.attn1(
                    norm_hidden_states, 
                    encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                    # encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,)

                if self.norm_type == "ada_norm_zero":
                    attn_output_uc = gate_msa.unsqueeze(1) * attn_output_uc
                elif self.norm_type == "ada_norm_single":
                    attn_output_uc = gate_msa * attn_output_uc

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
                        **cross_attention_kwargs,
                    )

                    if self.norm_type == "ada_norm_zero":
                        attn_output_uc_r = gate_msa[_uc_mask].unsqueeze(1) * attn_output_uc_r
                    elif self.norm_type == "ada_norm_single":
                        attn_output_uc_r = gate_msa[_uc_mask] * attn_output_uc_r

                    hidden_states_c[_uc_mask] = attn_output_uc_r + hidden_states[_uc_mask]

                    # hidden_states_c[_uc_mask] = self.attn1(
                    #     norm_hidden_states[_uc_mask],
                    #     encoder_hidden_states=norm_hidden_states[_uc_mask],
                    #     attention_mask=attention_mask,
                    # ) + hidden_states[_uc_mask]

                hidden_states = hidden_states_c.clone()

                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)
                # 1.2 GLIGEN Control
                if gligen_kwargs is not None:
                    hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

                self.bank.clear()
                # 3. Cross-Attention
                if self.attn2 is not None:
                    if self.norm_type == "ada_norm":
                        norm_hidden_states = self.norm2(hidden_states, timestep)
                    elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                        norm_hidden_states = self.norm2(hidden_states)
                    elif self.norm_type == "ada_norm_single":
                        # For PixArt norm2 isn't applied here:
                        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                        norm_hidden_states = hidden_states
                    elif self.norm_type == "ada_norm_continuous":
                        norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                    else:
                        raise ValueError("Incorrect norm")

                    if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                        norm_hidden_states = self.pos_embed(norm_hidden_states)

                    attn_output = self.attn2(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=encoder_attention_mask,
                        **cross_attention_kwargs,
                    )
                    hidden_states = attn_output + hidden_states

                # 4. Feed-forward
                # i2vgen doesn't have this norm 🤷‍♂️
                if self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
                elif not self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm3(hidden_states)

                if self.norm_type == "ada_norm_zero":
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

                if self.norm_type == "ada_norm_single":
                    norm_hidden_states = self.norm2(hidden_states)
                    norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

                if self._chunk_size is not None:
                    # "feed_forward_chunk_size" can be used to save memory
                    ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
                else:
                    ff_output = self.ff(norm_hidden_states)

                if self.norm_type == "ada_norm_zero":
                    ff_output = gate_mlp.unsqueeze(1) * ff_output
                elif self.norm_type == "ada_norm_single":
                    ff_output = gate_mlp * ff_output

                hidden_states = ff_output + hidden_states
                if hidden_states.ndim == 4:
                    hidden_states = hidden_states.squeeze(1)

                return hidden_states


        if self.reference_attn:
            if self.fusion_blocks == "midup":
                if self.model_type == "unet":
                    attn_modules = self.unet.transformer_blocks[self._refer_cross_layer_num:]
                if self.model_type == "controlnet":
                    attn_modules = self.unet.transformer_blocks[:self._control_cross_layer_num]
                if self.model_type == "appearencenet":
                    attn_modules = self.unet.transformer_blocks[self._refer_cross_layer_num:]
            elif self.fusion_blocks == "full":
                attn_modules = self.unet.transformer_blocks            
            # attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))
    
    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                if writer.model_type == "controlnet":
                    reader_attn_modules = self.unet.transformer_blocks[:self._control_cross_layer_num]
                    writer_attn_modules = writer.unet.transformer_blocks[:self._control_cross_layer_num]
                if writer.model_type == "appearencenet":
                    reader_attn_modules = self.unet.transformer_blocks[self._refer_cross_layer_num:]
                    writer_attn_modules = writer.unet.transformer_blocks[self._refer_cross_layer_num:]
                    
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
    
    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                if self.model_type == "unet":
                    reader_attn_modules = self.unet.transformer_blocks[self._refer_cross_layer_num:]
                if self.model_type == "controlnet":
                    reader_attn_modules = self.unet.transformer_blocks[:self._control_cross_layer_num]
                if self.model_type == "appearencenet":
                    reader_attn_modules = self.unet.transformer_blocks[self._refer_cross_layer_num:]
            elif self.fusion_blocks == "full":
                reader_attn_modules = self.unet.transformer_blocks
            # reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
            for r in reader_attn_modules:
                r.bank.clear()