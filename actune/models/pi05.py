"""Original ten-step OpenPI sampler, with host loop checks removed."""
import torch

@torch.no_grad()
def sample_actions_no_host_sync(self, device, observation, noise=None, num_steps=10):
    """Pinned OpenPI sample_actions with its ten-step while loop unrolled.

The GPU time tensor still starts at float32(1) and adds the same float32 dt
in place after every step. Do not replace this recurrence with a linspace,
Python float timesteps, fewer steps, or changed Euler arithmetic.
"""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    if num_steps != 10:
        raise ValueError("this exactness candidate supports the frozen ten-step sampler only")
    bsize = observation.state.shape[0]
    if noise is None:
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = self.sample_noise(actions_shape, device)
    images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
    self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_key_values = self.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
    dt = -1.0 / num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    for _ in range(num_steps):
        expanded_time = time.expand(bsize)
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t
        time += dt
    return x_t
