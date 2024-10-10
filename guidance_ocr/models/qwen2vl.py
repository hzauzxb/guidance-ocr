import torch
from transformers.cache_utils import DynamicCache

def prepare_model_first_inputs(model, **kwargs):
    model_inputs = {}
    for key in ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']:
        if key in kwargs.keys():
            model_inputs[key] = kwargs[key]
    
    attention_mask = torch.ones(model_inputs['input_ids'].shape, dtype = torch.int32).to(model_inputs['input_ids'].device)
    position_ids, rope_deltas = model.get_rope_index(
        model_inputs['input_ids'], model_inputs['image_grid_thw'], None, attention_mask
    )
    cache_position_ids = torch.max(position_ids)
    model_inputs['position_ids'] = position_ids
    model_inputs['use_cache'] = True
    model_inputs['past_key_values'] = DynamicCache()
    model_inputs['return_dict'] = True
    model_inputs['output_attentions'] = False
    model_inputs['output_hidden_states'] = False
    return model_inputs


def update_model_inputs(model_out, vaild_token_id, model_inputs, cache_position_ids, **kwargs):
    model_inputs['input_ids'] = torch.tensor([[vaild_token_id]]).to(model_inputs['input_ids'].device)
    model_inputs['past_key_values'] = model_out.past_key_values
    model_inputs['attention_mask'] = None
    model_inputs['pixel_values'] = None
    model_inputs['position_ids'] = torch.ones((3,1,1)).to(torch.int64).cuda() * cache_position_ids
    return model_inputs