"""Monkey-patch hooking the VIB adapter into Qwen2.5-VL's forward.

The base ``Qwen2_5_VLForConditionalGeneration.forward`` produces visual token
embeddings (``image_embeds = self.visual(...)``) and scatters them into
``inputs_embeds`` at image-token positions. We insert the bottleneck between
those two steps so the LLM consumes Z (the noisy reparameterized sample), not
the raw visual tokens. The patched forward is a faithful copy of the original;
the only behavioral change is guarded by ``getattr(self, "vib", None)``, so when
no VIB is attached the numerics are byte-for-byte identical to upstream.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
)

from open_r1.models.vib_adapter import VIBAdapter, VIBConfig


def get_vib_base_model(model) -> Qwen2_5_VLForConditionalGeneration:
    """Resolve the underlying Qwen2_5_VLForConditionalGeneration instance.

    The patched ``forward`` (and thus the VIB hook + ``_last_vib_rate``) runs on
    this inner module, even when it is wrapped by PEFT / DDP / DeepSpeed. The
    bottleneck state must be attached to and read from this same object.
    """
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        return model
    for module in model.modules():
        if isinstance(module, Qwen2_5_VLForConditionalGeneration):
            return module
    raise TypeError("No Qwen2_5_VLForConditionalGeneration found to attach VIB to.")


def attach_vib(model, vib_cfg: VIBConfig) -> Qwen2_5_VLForConditionalGeneration:
    """Build a VIBAdapter sized to the model, register it on the base Qwen module,
    and return that base module.

    Must be called before the optimizer is created so the VIB params are picked
    up by ``model.parameters()``.
    """
    base = get_vib_base_model(model)
    d_model = base.config.hidden_size
    param = next(base.parameters())
    adapter = VIBAdapter(
        d_model=d_model,
        cond_dim=vib_cfg.cond_dim,
        hidden_dim=vib_cfg.hidden_dim,
        n_cross_heads=vib_cfg.n_cross_heads,
        free_bits=vib_cfg.free_bits,
        logvar_min=vib_cfg.logvar_min,
        logvar_max=vib_cfg.logvar_max,
        instruction_conditioned=vib_cfg.instruction_conditioned,
        residual=vib_cfg.residual,
        suppress_eps=vib_cfg.suppress_eps,
        shared_noise_per_prompt=vib_cfg.shared_noise_per_prompt,
    )
    adapter = adapter.to(device=param.device, dtype=param.dtype)
    for p in adapter.parameters():
        p.requires_grad = True

    if vib_cfg.train_backbone:
        for p in base.parameters():
            p.requires_grad = True

    base.vib = adapter
    base.vib_cfg = vib_cfg
    # True during training and rollouts (noisy z). Eval may flip this to False.
    base.vib_sample = True
    base.vib_eval_mode = vib_cfg.eval_mode
    base.vib_eval_samples = vib_cfg.eval_samples
    base._last_vib_rate = None
    base._last_vib_stats = None
    base._vib_group_seed = None
    return base


def _resolve_sample_flag(model) -> bool:
    if model.training:
        return True
    # Eval: deterministic mean unless an explicit noisy eval mode is requested.
    return getattr(model, "vib_eval_mode", "mean") != "mean"


def _apply_vib(model, image_embeds, input_ids, inputs_embeds, attention_mask):
    """Apply the bottleneck per sample and return the replaced visual tokens.

    Qwen packs a variable number of image tokens per sample; image_embeds is the
    row-major concatenation over (batch, position) of image-token slots. We split
    it back per sample, condition on that sample's instruction text, compute the
    KL per sample, then average the rate over samples.
    """
    vib: VIBAdapter = model.vib
    image_token_id = model.config.image_token_id

    img_mask = input_ids == image_token_id  # [B, L]
    counts = img_mask.sum(dim=1).tolist()
    if attention_mask is not None:
        text_sel = attention_mask.bool() & (~img_mask)
    else:
        text_sel = ~img_mask

    sample_flag = _resolve_sample_flag(model)

    generator = None
    if vib.shared_noise_per_prompt and getattr(model, "_vib_group_seed", None) is not None:
        generator = torch.Generator(device=image_embeds.device)
        generator.manual_seed(int(model._vib_group_seed))

    z_chunks = []
    rates = []
    agg_stats = {
        "rate": [],
        "kl_token_mean": [],
        "suppressed_frac": [],
        "mu_abs_mean": [],
        "std_mean": [],
    }
    kl_per_token_list = []

    offset = 0
    B = input_ids.size(0)
    for b in range(B):
        n = int(counts[b])
        if n == 0:
            continue
        H_b = image_embeds[offset : offset + n].unsqueeze(0)  # [1, n, d]
        offset += n

        text_emb_b = inputs_embeds[b][text_sel[b]].unsqueeze(0)  # [1, Lt, d]
        text_mask_b = torch.ones(
            text_emb_b.shape[:2], dtype=torch.bool, device=text_emb_b.device
        )
        Z_b, rate_b, stats_b = vib(
            H_b, text_emb_b, text_mask_b, sample=sample_flag, generator=generator
        )
        z_chunks.append(Z_b.squeeze(0))
        rates.append(rate_b)
        for k in agg_stats:
            agg_stats[k].append(stats_b[k])
        kl_per_token_list.append(stats_b["kl_per_token"].squeeze(0))

    if len(z_chunks) == 0:
        model._last_vib_rate = None
        model._last_vib_stats = None
        return image_embeds

    new_image_embeds = torch.cat(z_chunks, dim=0).to(image_embeds.dtype)
    rate = torch.stack(rates).mean()  # mean over samples; keeps autograd graph
    model._last_vib_rate = rate
    model._last_vib_stats = {
        k: torch.stack(v).mean() for k, v in agg_stats.items()
    }
    model._last_vib_stats["kl_per_token"] = kl_per_token_list
    return new_image_embeds


def pop_vib_rate(self) -> Optional[torch.Tensor]:
    """Return the last realized rate (R_hat) and clear it.

    The trainer calls this right after the policy forward to add ``beta * R_hat``
    to the loss. Returns None when VIB is disabled or no rate was produced.
    """
    rate = getattr(self, "_last_vib_rate", None)
    self._last_vib_rate = None
    return rate


def pop_vib_stats(self) -> Optional[dict]:
    stats = getattr(self, "_last_vib_stats", None)
    self._last_vib_stats = None
    return stats


def set_vib_group_seed(self, seed: Optional[int]):
    self._vib_group_seed = seed


def vib_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)

            # ---- VIB hook: replace raw visual tokens with the bottleneck sample ----
            if getattr(self, "vib", None) is not None:
                image_embeds = _apply_vib(
                    self, image_embeds, input_ids, inputs_embeds, attention_mask
                )
            # ----------------------------------------------------------------------

            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        logits = logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )


_PATCHED = False


def patch_qwen2_5_vl_with_vib():
    """Install the VIB-aware forward and helper methods (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    Qwen2_5_VLForConditionalGeneration.forward = vib_forward
    Qwen2_5_VLForConditionalGeneration.pop_vib_rate = pop_vib_rate
    Qwen2_5_VLForConditionalGeneration.pop_vib_stats = pop_vib_stats
    Qwen2_5_VLForConditionalGeneration.set_vib_group_seed = set_vib_group_seed
    _PATCHED = True
