import torch
from transformers.cache_utils import DynamicCache

def prepare_model_first_inputs(model, **kwargs):
    model_inputs = {}
    model_inputs['input_ids'] = kwargs['input_ids']
    model_inputs['use_cache'] = True
    model_inputs['past_key_values'] = None
    model_inputs['return_dict'] = True
    model_inputs['output_attentions'] = False
    model_inputs['output_hidden_states'] = False
    return model_inputs


def update_model_inputs(model_out, vaild_token_id, model_inputs, cache_position_id, **kwargs):
    model_inputs['past_key_values'] = model_out.past_key_values
    model_inputs['input_ids'] = torch.tensor([[vaild_token_id]]).to(model_inputs['input_ids'].device)
    model_inputs['position_ids'] = torch.tensor([[cache_position_id]]).to(model_inputs['input_ids'].device)
    return model_inputs