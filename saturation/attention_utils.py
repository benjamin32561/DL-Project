import math
import torch
import pdb

from constants import OUT_INDEX, STYLE_INDEX, STRUCT_INDEX


def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    """ Verify whether we should perform the mixing in the current timestep. """
    is_in_32_timestep_range = (
            model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end
    )
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end
    )
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or \
                 (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix


def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False, contrast_strength=1.0):
    '''Compute the scale dot product attention, potentially with our contrasting operation.'''
    atten_map   = (Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1)))
    attn_weight = torch.softmax(atten_map, dim=-1)
    if edit_map and not is_cross:

      attn_weight[OUT_INDEX] = torch.stack([
          torch.clip(entropy_equalization(attn_weight[OUT_INDEX][head_idx],
                                          attn_weight[STYLE_INDEX][head_idx],
                                          attn_weight[STRUCT_INDEX][head_idx]),min=0.0, max=1.0)
          for head_idx in range(attn_weight.shape[1])
        ])
    return attn_weight @ V, attn_weight


def entropy_equalization(tensor_out: torch.Tensor, tensor_app: torch.Tensor,tensor_str: torch.Tensor) -> torch.Tensor:
    """ Equalize attention map using entropy. """
   
    entropy_out = compute_entropy(tensor_out)
    Beta   = (1 + 0.025*(entropy_out))
    Beta_T = Beta.unsqueeze(1)
    tensor_out_device = tensor_out.device
    Beta_T = Beta_T.to(tensor_out_device)
    adjusted_tensor = (tensor_out - tensor_out.mean(dim=-1)) * Beta_T + tensor_out.mean(dim=-1)
    return adjusted_tensor

 
def compute_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy for each row of the attention weights.
    """
    eps = 1e-9  # Small epsilon to avoid log(0)
    attention_weights = attention_weights + eps
    entropy = -torch.sum(attention_weights * torch.log2(attention_weights), dim=-1)
    return entropy
