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

class OCRProcessor(object):
    def __init__(
        self, ocr_tree, tokenizer, 
        top_k, eos_id
    ):
        self.ocr_tree = ocr_tree
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.eos_id = eos_id

        self.ocr_all_nodes = self.ocr_tree.get_init_avinodes(self.ocr_tree.root)
        self.ocr_head_nodes = self.ocr_tree.get_head_nodes()
        self.avi_nodes = self.ocr_all_nodes

        self.filter_value = -float('inf')


    def __call__(self, input_ids, scores):
        order_indexes = scores.argsort(dim = -1, descending = True)[..., : self.top_k]
        order_indexes = order_indexes[0]

        valid_token = self.eos_id

        for token in order_indexes:
            text = self.tokenizer.decode(token)
            is_value_vaild, possible_avi_nodes = is_valid(self.avi_nodes, text, self.ocr_head_nodes)
            if is_value_vaild:
                self.avi_nodes = possible_avi_nodes
                valid_token = token
                break

            if token == self.eos_id:
                valid_token = token
                break

    

        # 结束时重置状态
        if valid_token == self.eos_id:
            self.avi_nodes = self.ocr_all_nodes

        vocab_size = scores.size()[-1]
        mask = torch.ones((vocab_size), dtype = torch.bool, device = scores.device)
        mask[valid_token] = 0
        scores_processed = scores.masked_fill(mask, self.filter_value)
        return scores_processed

def get_ocr_processor(
    text_list, tokenizer, 
    top_k, eos_id
):
    ocr_tree = init_trie(text_list)
    return OCRProcessor(
        ocr_tree, tokenizer, top_k, eos_id = eos_id
    )