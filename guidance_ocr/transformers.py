# 对transformers模型进行改造，使用ocr结果完善输出内容
import torch
import inspect
import numpy as np
import transformers
from types import MethodType

from guidance_ocr.tire import Trie

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

def ocrguid_generate(self, **kwargs):
    # 用于存储当前可用的所有state
    tire_avi_nodes = self.tire.get_init_avinodes(self.tire.root)
    self.tire_init_nodes = self.tire.get_init_avinodes(self.tire.root)

    model_inputs = {}
    for key in kwargs.keys():
        if key in self.forward_paras:
            model_inputs[key] = kwargs[key]

    model_inputs['use_cache'] = True
    model_inputs['past_key_values'] = None
    model_inputs['return_dict'] = True
    model_inputs['output_attentions'] = False
    model_inputs['output_hidden_states'] = False
    
    import pdb; pdb.set_trace()

    output_ids = kwargs['input_ids'][0].cpu().numpy().tolist()

    for _ in range(kwargs['max_new_tokens']):
        with torch.no_grad():
            model_out = self.forward(**model_inputs)
            logits = model_out['logits'][0][-1].cpu().numpy()
            token_ids = np.argsort(-logits)
            
            vaild_token_id = -1
            token_cnt = 0
            for token_id in token_ids:
                token_cnt += 1
                if token_cnt >= 100:
                    break
                text = self.tokenizer.decode(token_id)
                valid_res, avi_nodes = is_valid(token_id, tire_avi_nodes, text)
                if valid_res:
                    tire_avi_nodes = avi_nodes
                    if len(avi_nodes) == 0:
                        tire_avi_nodes = self.tire_init_nodes
                  
                    vaild_token_id = token_id
                    break

            if vaild_token_id == -1:
                break
            output_ids.append(vaild_token_id)
            model_inputs['past_key_values'] = model_out.past_key_values
            model_inputs['input_ids'] = torch.tensor([[vaild_token_id]]).to(self.device)
            model_inputs['position_ids'] = torch.tensor([[len(output_ids)-1]]).to(self.device)
            out_text = self.tokenizer.decode(vaild_token_id)
            if out_text == '<|im_end|>':
                break

            

    return [output_ids]

def warp_tfmodel(model, tokenizer, text_list):
    class Warp(object):
        def __enter__(self):
            # 解析forward函数的传参
            forward_obj = inspect.signature(model.forward)
            model.forward_paras = forward_obj.parameters.keys()
            model.tokenizer = tokenizer
            model.tire = init_trie(text_list)
            model.generate = MethodType(ocrguid_generate, model)
            return model

        def __exit__(self, exc_type, exc_val, exc_tb):
            return
    return Warp()