import os
import torch
import copy
import hashlib
import warnings
from typing import Any, List, Optional, Tuple, Union

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast


VISION_END_ID = 92545

def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict

        if pixel_values is not None:
            input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

            vit_embeds = self.extract_feature(pixel_values)
            vit_batch_size = pixel_values.shape[0]

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

       
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)

            input_embeds = input_embeds.reshape(B, N, C)
            outputs = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            outputs = self.language_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits

            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

def normal_first_inputs(model, **kwargs):
    model_inputs = {}
    for key in ['input_ids', 'pixel_values']:
        if key in kwargs.keys():
            model_inputs[key] = kwargs[key]
    model_inputs['use_cache'] = True
    model_inputs['return_dict'] = True
    model_inputs['output_attentions'] = False
    model_inputs['output_hidden_states'] = False

    return model_inputs

def accelerate_first_inputs(model, max_cache, **kwargs):
    input_ids = kwargs['input_ids']
    vision_end_index = int(torch.where(input_ids==VISION_END_ID)[1][0].cpu().numpy())
    cache_input_ids = input_ids[:,:vision_end_index]
    cache_text = model.tokenizer.decode(cache_input_ids[0]).replace(' ', '')
    img_hash = hashlib.sha256(kwargs['pixel_values'].cpu().float().numpy()).hexdigest()
    cache_key = cache_text + '_' + img_hash

    if cache_key not in model.kvcache_queue.keys():
        cache_inputs = dict(
            input_ids = cache_input_ids,
            pixel_values = kwargs['pixel_values'],
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
    

    model_inputs = dict(
        input_ids = kwargs['input_ids'][..., vision_end_index + 1:],
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
    if torch.sum(ind) != 1 and accelerate:
        warnings.warn('accelerate=True does not support multi img input, set accelerate=False')
        accelerate = False

    if accelerate == False:
        return normal_first_inputs(model, **kwargs)
    else:
        return accelerate_first_inputs(model, max_cache, **kwargs)

def update_model_inputs(model_out, vaild_token_id, model_inputs, cache_position_ids, **kwargs):
    model_inputs['input_ids'] = torch.LongTensor([[vaild_token_id]]).to(model_inputs['input_ids'].device)
    model_inputs['past_key_values'] = model_out.past_key_values
    model_inputs['attention_mask'] = None
    model_inputs['pixel_values'] = None
    model_inputs['position_ids'] = None# torch.LongTensor([[cache_position_ids]]).to(model_inputs['input_ids'].device)
    # import pdb; pdb.set_trace()
    return model_inputs