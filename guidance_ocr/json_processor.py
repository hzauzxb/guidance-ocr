# 在generate函数中加入logit_processor即可完成对模型输出的规范化， 支持输出json
from typing import List, Dict, Set, Tuple, Optional
from queue import Queue

import torch
from transformers import AutoTokenizer

from .trie import Trie

def init_trie(text_list):
    trie = Trie()
    for text in text_list:
        trie.insert(text)
    return trie

def is_valid_node(text, node):
    if node.char != text[0]:
        return False, None, False
    
    if len(text) == 1:
        avi_nodes = []
        for children_key in node.children:
            avi_nodes.append(node.children[children_key]) 
        return True, avi_nodes, node.is_end_of_word

    is_valid = False; avi_nodes = None; is_end_of_word = False
    for sub_node_key in node.children.keys():
        sub_is_valid, sub_avi_nodes, sub_is_end_of_word = is_valid_node(text[1:], node.children[sub_node_key])
        if sub_is_valid:
            is_valid = True
            avi_nodes = sub_avi_nodes
            is_end_of_word = sub_is_end_of_word
            break
        
    return is_valid, avi_nodes, is_end_of_word
        
def is_valid(avi_nodes, text, all_head_nodes):
    # 判断输出是否合法
    is_valid = False; possible_avi_nodes = []
    end_flag = 0
    for node in avi_nodes:
        sub_is_valid, sub_avi_nodes, is_end_of_word = is_valid_node(text, node)
        if sub_is_valid:
            is_valid = True

            if is_end_of_word and end_flag == 0:
                possible_avi_nodes += all_head_nodes
                end_flag = 1
            possible_avi_nodes += sub_avi_nodes
    
    return is_valid, possible_avi_nodes

class JsonProcessor(object):
    def __init__(
        self, ocr_tree, key_tree, special_charset, 
        tokenizer, top_k, eos_id
    ):
        self.ocr_tree = ocr_tree
        self.key_tree = key_tree
        self.special_charset = special_charset
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.eos_id = eos_id

        self.key_all_nodes = self.key_tree.get_init_avinodes(self.key_tree.root)
        self.ocr_all_nodes = self.ocr_tree.get_init_avinodes(self.ocr_tree.root)

        self.key_head_nodes = self.key_tree.get_head_nodes()
        self.ocr_head_nodes = self.ocr_tree.get_head_nodes()

        self.avi_nodes = None

        self.SPECIAL_STATUS = 0
        self.OCR_STATUS = 1
        self.KEY_STATUS = 2
        self.status = self.SPECIAL_STATUS

        self.SPECIAL2KEY = True
        self.SPECIAL2VALUE = False

        self.filter_value = -float('inf')


    def __call__(self, input_ids, scores):
        order_indexes = scores.argsort(dim = -1, descending = True)[..., : self.top_k]

        # 为了vllm做的兼容，vllm中logit_processor传的score是一维的
        # transformers中，logit_processor的score是二维的
        if len(order_indexes) == 1:
            order_indexes = order_indexes[0]

        valid_token = self.eos_id

        for token in order_indexes:
            if self.status == self.SPECIAL_STATUS:
                # 判断token是否在特殊字符中
                if token in self.special_charset:
                    valid_token = token
                    break
                text = self.tokenizer.decode(token)
                
                # 判断token是否在 key 中
                if self.SPECIAL2KEY:
                    is_key_valid, possible_avi_nodes = is_valid(self.key_all_nodes, text, self.key_head_nodes)
                    if is_key_valid:
                        self.status = self.KEY_STATUS
                        self.avi_nodes = possible_avi_nodes
                        valid_token = token
                        break

                # 判断token是否在ocr中
                if self.SPECIAL2VALUE:
                    is_ocr_valid, possible_avi_nodes = is_valid(self.ocr_all_nodes, text, self.ocr_head_nodes)
                    
                    if is_ocr_valid:
                        self.status = self.OCR_STATUS
                        self.avi_nodes = possible_avi_nodes
                        valid_token = token
                        break
                
                if token == self.eos_id:
                    break
            elif self.status == self.KEY_STATUS:
                # key 不会出现换行的情况只需从头找到尾就好
                if len(self.avi_nodes) == 0:    # 找到尾了，需要从sepcial中来找
                    if token in self.special_charset:
                        valid_token = token
                        self.status = self.SPECIAL_STATUS
                        self.SPECIAL2VALUE = True
                        self.SPECIAL2KEY = False
                        break

                text = self.tokenizer.decode(token)
                is_key_valid, possible_avi_nodes = is_valid(self.avi_nodes, text, [])
                if is_key_valid:
                    self.avi_nodes = possible_avi_nodes
                    valid_token = token
                    break
            elif self.status == self.OCR_STATUS:
                
                # 判断是否在value中
                text = self.tokenizer.decode(token)
                is_value_vaild, possible_avi_nodes = is_valid(self.avi_nodes, text, self.ocr_head_nodes)
                if is_value_vaild:
                    self.avi_nodes = possible_avi_nodes
                    valid_token = token
                    break

                if token in self.special_charset:
                    valid_token = token
                    self.status = self.SPECIAL_STATUS
                    self.SPECIAL2VALUE = False
                    self.SPECIAL2KEY = True
                    break



        # 结束时重置状态
        if valid_token == self.eos_id:
            self.SPECIAL2KEY = True
            self.SPECIAL2VALUE = False

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
        special_charset = [tokenizer(c)['input_ids'][0] for c in special_charset]
        if eos_id is None:
            eos_id = tokenizer(eos_text)['input_ids'][0]
    elif model_type == 'internvl2':
        # [364,   647, 387]
        special_charset = ['{\n', '  ', ' \"', '\":', '\"', '\",\n', '}\n', '}', '\"\n', '\",', '\n']
        special_charset = [tokenizer(c)['input_ids'][1] for c in special_charset]
        if eos_id is None:
            eos_id = tokenizer(eos_text)['input_ids'][1]

    return JsonProcessor(
        ocr_tree, key_tree,
        special_charset, tokenizer, top_k, eos_id = eos_id
    )