# 在generate函数中加入logit_processor即可完成对模型输出的规范化， 支持输出json
import re
import torch
from queue import Queue
from typing import List, Dict, Set, Tuple, Optional

import torch
from transformers import AutoTokenizer

from .trie import Trie

def init_trie(text_list):
    trie = Trie()
    for text in text_list:
        trie.insert(text)
    return trie


def get_valid_nodes_from_root(generated_text, root_node, head_nodes):
    avi_nodes = []
    use_all_nodes = False

    if len(generated_text) == 0:
        if root_node.char is None: # root节点且无输入的话，可以使用全部节点
            return True, []
        else:
            return False, [root_node.children[key] for key in root_node.children.keys()]

    # 在子节点中找
    for node_key in root_node.children.keys():
        node = root_node.children[node_key]
        if node.char == generated_text[0]:
            use_all_nodes, sub_avi_nodes = get_valid_nodes_from_root(generated_text[1:], node, head_nodes)
            if use_all_nodes:
                return True, []
            avi_nodes = avi_nodes + sub_avi_nodes
       
    # 如果是字符串末尾节点还可以在头节点中找
    if root_node.is_end_of_word:
        for node in head_nodes:
            if node.char == generated_text[0]:
                use_all_nodes, sub_avi_nodes = get_valid_nodes_from_root(generated_text[1:], node, head_nodes)
                if use_all_nodes:
                    return True, []
                avi_nodes = avi_nodes + sub_avi_nodes
    return False, avi_nodes

def get_valid_nodes(generated_text, all_nodes, head_nodes):
    valid_nodes = []
    if len(generated_text) == 0:
        return all_nodes

    for node in all_nodes:
        if node.char is not None and node.char == generated_text[0]:
            use_all_nodes, sub_valid_nodes = get_valid_nodes_from_root(
                generated_text[1:], node, head_nodes
            )
            if use_all_nodes:
                return all_nodes
            else:
                valid_nodes += sub_valid_nodes
    return valid_nodes
    
def is_valid(text, node, head_nodes):
    if len(text) == 0:
        return True

    if node.char != text[0]:
        return False

    sub_nodes = [node.children[key] for key in node.children.keys()]
    if node.is_end_of_word:
        sub_nodes += head_nodes

    for sub_node in sub_nodes:
        if is_valid(text[1:], sub_node, head_nodes):
            return True

    return False

class JsonProcessor(object):
    def __init__(
        self, ocr_tree, key_tree, special_charset, 
        tokenizer, top_k, eos_id, gen_start_text
    ):
        self.ocr_tree = ocr_tree
        self.key_tree = key_tree
        self.special_charset = special_charset
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.eos_id = eos_id
        self.gen_start_text = gen_start_text

        self.key_all_nodes = self.key_tree.get_init_avinodes(self.key_tree.root)
        self.ocr_all_nodes = self.ocr_tree.get_init_avinodes(self.ocr_tree.root)

        self.key_head_nodes = self.key_tree.get_head_nodes()
        self.ocr_head_nodes = self.ocr_tree.get_head_nodes()

        self.JSON_START = 0
        self.KEY = 1
        self.JSON_KV = 2
        self.VAL = 3
        self.JSON_VKEND = 4
       

        self.filter_value = -float('inf')

    def get_cur_state(self, generated_text):
        count = generated_text.count("\"")
        if count == 0:
            return [self.JSON_START]

        double_quotes_cnt = re.findall("\"(.*?)\"", generated_text)
        
        if len(double_quotes_cnt) % 2 == 0:
            if count % 2 == 1:
                return [self.KEY, self.JSON_KV]
            else:
                return [self.JSON_VKEND]
        else:
            if count % 2 == 1:
                return [self.VAL, self.JSON_VKEND]
            else:
                return [self.JSON_KV]


    def __call__(self, input_ids, scores):
        if type(input_ids) == torch.Tensor:
            input_text = self.tokenizer.decode(input_ids[0])
        else:
            input_text = self.tokenizer.decode(input_ids)
        generated_text = input_text.split(self.gen_start_text)[-1]

        # 兼容投机采样，根据已生成的token判断当前状态
        self.status = self.get_cur_state(generated_text)
        order_indexes = scores.argsort(dim = -1, descending = True)[..., : self.top_k]

        # 兼容vllm 离线部署
        if len(order_indexes) == 1:
            order_indexes = order_indexes[0]

        # 如果存在key和value，需要预先计算可行的token从而节省时间
        valid_nodes = []
        if self.KEY in self.status:
            key_text = generated_text.split('\"')[-1]
            valid_nodes = get_valid_nodes(key_text, self.key_all_nodes, self.key_head_nodes)
        if self.VAL in self.status:
            count = generated_text.count("\"")
            if count %2 == 0:
                val_text = ''
            else:
                val_text = generated_text.split('\"')[-1]
            valid_nodes = get_valid_nodes(val_text, self.ocr_all_nodes, self.ocr_head_nodes)

        # 按logit从大到小遍历候选token
        valid_token = self.eos_id
        for token in order_indexes:
            if self.JSON_START in self.status and \
                 token in self.special_charset['json_start']:
                valid_token = token

            token_text = self.tokenizer.decode(token)
            if self.KEY in self.status:
                for valid_node in valid_nodes:
                    if is_valid(token_text, valid_node, self.key_head_nodes):
                        valid_token = token
                        break
            
            if self.JSON_KV in self.status and\
                 token in self.special_charset['json_kv']:
                valid_token = token

            if self.VAL in self.status:
                for valid_node in valid_nodes:
                    if is_valid(token_text, valid_node, self.ocr_head_nodes):
                        valid_token = token
                        break

            if self.JSON_VKEND in self.status and \
                 token in self.special_charset['json_vkend']:
                valid_token = token
                

            if valid_token != self.eos_id:
                break

        # 根据合法的token更改scores
        vocab_size = scores.size()[-1]
        mask = torch.ones((vocab_size), dtype = torch.bool, device = scores.device)
        mask[valid_token] = 0
        scores_processed = scores.masked_fill(mask, self.filter_value)
        return scores_processed

def get_json_processor(
    text_list, extract_keys, 
    tokenizer, top_k, model_type, eos_text = None, eos_id = None
):
    assert eos_text is not None or eos_id is not None

    ocr_tree = init_trie(text_list)
    key_tree = init_trie(extract_keys)

    if model_type == 'qwen2vl':
        special_charset = ['{\n', '}', ' :', '\n', ' ', ' \"', '\"', '\",\n', '\"\n']
        special_charset = dict(
            json_start = [tokenizer(c)['input_ids'][0] for c in ["{\n", " \"", "\"", " "]],
            json_kv = [tokenizer(c)['input_ids'][0] for c in ["\":", " \"", "\""]],
            json_vkend = [tokenizer(c)['input_ids'][0] for c in [" \"", "\",\n", "\"\n","}", " "]],
        )
        if eos_id is None:
            eos_id = tokenizer(eos_text)['input_ids'][0]
        gen_start_text = 'assistant\n'
    else:
        raise NotImplementedError

    return JsonProcessor(
        ocr_tree, key_tree,
        special_charset, tokenizer, top_k, 
        eos_id = eos_id, gen_start_text = gen_start_text
    )