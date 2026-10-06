"""Original OFT action-only prediction path; leaves the action head intact."""
import torch

class UnusedVocabularyProjection(torch.nn.Module):
    def forward(self, *_args, **_kwargs):
        raise RuntimeError("vocabulary projection was released for the action-only runtime")

def _action_only_prediction(vla, input_embeddings, all_actions_mask, projected_patch_embeddings,
                            attention_mask, labels, NUM_PATCHES, NUM_PROMPT_TOKENS, action_head=None):
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
    if action_head is None:
        raise ValueError("action-only runtime requires the L1 regression head")
    all_actions_mask = all_actions_mask.unsqueeze(-1)
    input_embeddings = input_embeddings * ~all_actions_mask
    embeddings, mask = vla._build_multimodal_attention(input_embeddings, projected_patch_embeddings, attention_mask)
    output = vla.language_model.model(input_ids=None, attention_mask=mask, position_ids=None,
        past_key_values=None, inputs_embeds=embeddings, use_cache=False,
        output_attentions=False, output_hidden_states=False, return_dict=True)
    actions_hidden_states = output.last_hidden_state[
        :, NUM_PATCHES + NUM_PROMPT_TOKENS:NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK, :]
    normalized_actions = action_head.predict_action(actions_hidden_states)
    normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)
    normalized_actions = normalized_actions.float().cpu().detach().numpy()
    return normalized_actions, actions_hidden_states
