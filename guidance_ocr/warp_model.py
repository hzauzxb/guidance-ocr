# 对transformers模型进行改造，使用ocr结果完善输出内容
import copy
import torch
import warnings
import numpy as np
import transformers
from types import MethodType
from transformers.cache_utils import DynamicCache

from .trie import Trie

from .models import prepare_model_first_inputs_map, update_model_inputs_map, \
                    eos_text_map, forward_map

def init_trie(text_list):
    trie = Trie()
    for text in text_list:
        trie.insert(text)
    return trie

# 从该node开始找text是否合法，若合法则需要返回 后续可用的node，使用递归实现
def is_valid_node(text, node):
    if node.char != text[0]:
        return False, None
    
    if len(text) == 1:
        avi_nodes = []
        for children_key in node.children:
            avi_nodes.append(node.children[children_key]) 
        return True, avi_nodes

    is_valid = False; avi_nodes = None
    for sub_node_key in node.children.keys():
        sub_is_valid, sub_avi_nodes = is_valid_node(text[1:], node.children[sub_node_key])
        if sub_is_valid:
            is_valid = True
            avi_nodes = sub_avi_nodes
            break
        
    return is_valid, avi_nodes
        
def is_valid(tire, avi_nodes, text):
    # 判断输出是否合法
    is_valid = False; possible_avi_nodes = []
    for node in avi_nodes:
        sub_is_valid, sub_avi_nodes = is_valid_node(text, node)
        if sub_is_valid:
            is_valid = True
            possible_avi_nodes += sub_avi_nodes
    return is_valid, possible_avi_nodes


def get_ocrguid_generate_func(model_type, allow_texts, top_k, accelerate, max_cache):
    assert model_type in ['qwen2vl', 'internvl2']
    def ocrguid_generate(self, **kwargs):
        # 用于存储当前可用的所有state
        tire_avi_nodes = self.tire.get_init_avinodes(self.tire.root)
        self.tire_init_nodes = self.tire.get_init_avinodes(self.tire.root)

        eos_text = eos_text_map[model_type]
        allow_texts_all = allow_texts + eos_text

        # 获取forward函数的首次输入
        prepare_model_first_inputs = prepare_model_first_inputs_map[model_type]
        model_inputs = prepare_model_first_inputs(self, accelerate, max_cache, **kwargs)
        if 'position_ids' in model_inputs.keys():                
            cache_position_id = torch.max(model_inputs['position_ids'])
        else:
            cache_position_id = model_inputs['input_ids'].shape[1] - 1

        update_model_inputs = update_model_inputs_map[model_type]
        output_ids = []
        for _ in range(kwargs['max_new_tokens']):
            with torch.no_grad():
                # 获取模型输出概率分布
                model_out = self.forward(**model_inputs)
                logits = model_out['logits'][0][-1].cpu().numpy()
                token_ids = np.argsort(-logits)
                
                # 找首个满足要求的token
                vaild_token_id = -1
                token_cnt = 0
                for token_id in token_ids:
                    token_cnt += 1
                    if token_cnt >= top_k:
                        break
                    text = self.tokenizer.decode(token_id).replace(' ', '')
                    if len(text) == 0:
                        valid_res = 0; avi_nodes = []
                    else:
                        valid_res, avi_nodes = is_valid(token_id, tire_avi_nodes, text)
                    if valid_res:
                        tire_avi_nodes = avi_nodes
                        if len(avi_nodes) == 0:
                            tire_avi_nodes = self.tire_init_nodes
                    
                        vaild_token_id = token_id
                        break
                    elif text in allow_texts_all:
                        vaild_token_id = token_id
                        tire_avi_nodes = self.tire_init_nodes
                        break

                # 找不到则停止输出
                if vaild_token_id == -1:
                    break
                
                output_ids.append(vaild_token_id)

                cache_position_id += 1
                model_inputs = update_model_inputs(model_out, vaild_token_id, model_inputs, cache_position_id)

                out_text = self.tokenizer.decode(vaild_token_id).replace(' ', '')

                if out_text in eos_text:
                    break

        return [output_ids]
    return ocrguid_generate

# 修改模型的generate函数
def warp_model(model, tokenizer, text_list, model_type, allow_texts = [], top_k = 100, accelerate = True, max_cache=20):
    class Warp(object):
        def __enter__(self):
            self.model_type = model_type
            # 根据单词获取字典树
            new_text_list = []
            for t in text_list:
                if ' ' in t:
                    tt = t.replace(' ', '')
                    warnings.warn(f'space is not allowed in text list, replace\"{t}\" with \"tt\"')
                    new_text_list.append(tt)
                else:
                    new_text_list.append(t)

            model.tire = init_trie(new_text_list)

            # 替换模型的generate函数
            model.tokenizer = tokenizer
            ocrguid_generate = get_ocrguid_generate_func(model_type, allow_texts, top_k, accelerate, max_cache) # 根据模型type重新获取generate函数

            self.model = model            
            self.model.cache_generate = model.generate

            if model_type in forward_map.keys():
                self.model.cache_forward = model.forward
                self.model.forward = MethodType(forward_map[model_type], model)

            self.model.generate = MethodType(ocrguid_generate, model)

            return self.model

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.model_type == 'internvl2':
                self.model.forward = self.model.cache_forward
            self.model.generate = self.model.cache_generate

    return Warp()