import torch
import copy
import hashlib
import warnings
from transformers.cache_utils import DynamicCache

VISION_END_ID = 151653

def normal_first_inputs(model, **kwargs):
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

def accelerate_first_inputs(model, max_cache, **kwargs):
    input_ids = kwargs['input_ids']
    vision_end_index = int(torch.where(input_ids==VISION_END_ID)[1][0].cpu().numpy())

    cache_input_ids = input_ids[:,:vision_end_index]
    cache_text = model.tokenizer.decode(cache_input_ids[0])
    img_hash = hashlib.sha256(kwargs['pixel_values'].cpu().numpy()).hexdigest()
    cache_key = cache_text + '_' + img_hash

    if cache_key not in model.kvcache_queue.keys():
        cache_inputs = dict(
            input_ids = cache_input_ids,
            pixel_values = kwargs['pixel_values'],
            image_grid_thw = kwargs['image_grid_thw'],
        )
        cache_inputs = normal_first_inputs(model, **cache_inputs)
        with torch.no_grad():
            cache_output = model.forward(**cache_inputs)
        cache_kv_cache = cache_output.past_key_values

        if len(model.kvcache_queue) >= max_cache:
            del_list = [[key, -model.kvcache_queue[key][1]] for key in model.kvcache_queue.keys()]
            del_list.sort(key = lambda x:x[1])
            del_list = del_list[:len(model.kvcache_queue) - (max_cache -1)]
            for del_key in del_list:
                model.kvcache_queue.pop(del_key[0])
                # print(-del_key[1])
        model.kvcache_queue[cache_key] = [cache_kv_cache, 0]

    cache_kv_cache = copy.deepcopy(model.kvcache_queue[cache_key][0])
    model.kvcache_queue[cache_key][1] += 1
    
    attention_mask = torch.ones(kwargs['input_ids'].shape, dtype = torch.int32).to(cache_input_ids.device)
    position_ids, rope_deltas = model.get_rope_index(
        kwargs['input_ids'], kwargs['image_grid_thw'], None, attention_mask
    )

    model_inputs = dict(
        input_ids = kwargs['input_ids'][..., vision_end_index + 1:],
        attention_mask = None,
        position_ids = position_ids[..., vision_end_index + 1:],
        use_cache = True,
        past_key_values = cache_kv_cache,
        return_dict = True,
        output_attentions = False,
        output_hidden_states = False,
        pixel_values = None
    )
    return model_inputs

def prepare_model_first_inputs(model, accelerate, max_cache, **kwargs):
    try:
        kvcache_queue = model.kvcache_queue
    except:
        model.kvcache_queue = {}

    ind = kwargs['input_ids'] == VISION_END_ID
    if torch.sum(ind) != 1:
        warnings.warn('accelerate=True does not support multi img input, set accelerate=False')
        accelerate = False

    if accelerate == False:
        return normal_first_inputs(model, **kwargs)
    else:
        return accelerate_first_inputs(model, max_cache, **kwargs)


def update_model_inputs(model_out, valid_token_id, model_inputs, cache_position_ids, **kwargs):
    model_inputs['input_ids'] = torch.tensor([[valid_token_id]]).to(model_inputs['input_ids'].device)
    model_inputs['past_key_values'] = model_out.past_key_values
    model_inputs['attention_mask'] = None
    model_inputs['pixel_values'] = None
    model_inputs['position_ids'] = torch.ones((3,1,1)).to(torch.int64).cuda() * cache_position_ids
    return model_inputs