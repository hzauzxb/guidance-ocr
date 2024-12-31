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


def get_valid_nodes_from_root(generated_text, root_node, head_nodes):
    avi_nodes = []
    use_all_nodes = False

    if len(generated_text) == 0:
        if root_node.char is None: # root节点且无输入的话，可以使用全部节点
            return True, []
        else:
            if root_node.is_end_of_word:
                return False, [root_node.children[key] for key in root_node.children.keys()] + head_nodes
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



class OCRProcessor(object):
    def __init__(
        self, ocr_tree, tokenizer, 
        gen_start_text, top_k, eos_id
    ):
        self.ocr_tree = ocr_tree
        self.tokenizer = tokenizer

        self.gen_start_text = gen_start_text
        self.top_k = top_k
        self.eos_id = eos_id

        self.ocr_all_nodes = self.ocr_tree.get_init_avinodes(self.ocr_tree.root)
        self.ocr_head_nodes = self.ocr_tree.get_head_nodes()

        self.filter_value = -float('inf')


    def __call__(self, input_ids, scores):
        # 获取已生成的字符串
        if type(input_ids) == torch.Tensor: # 使用transformers
            input_text = self.tokenizer.decode(input_ids[0])
        else: # 使用vllm >= 0.6.5
            input_text = self.tokenizer.decode(input_ids)
        generated_text = input_text.split(self.gen_start_text)[-1]

        # 根据已生成的字符串获取可行的节点 
        valid_nodes = get_valid_nodes(generated_text, self.ocr_all_nodes, self.ocr_head_nodes)

        order_indexes = scores.argsort(dim = -1, descending = True)[..., : self.top_k]
        if len(order_indexes) == 1: # 兼容vllm 离线部署
            order_indexes = order_indexes[0]

        valid_token = self.eos_id
        for token in order_indexes:
            token_text = self.tokenizer.decode(token)
            for valid_node in valid_nodes:
                if is_valid(token_text, valid_node, self.ocr_head_nodes):
                    valid_token = token
                    break
           
            if token == self.eos_id:
                valid_token = token
                break

            if valid_token != self.eos_id:
                break

        vocab_size = scores.size()[-1]
        mask = torch.ones((vocab_size), dtype = torch.bool, device = scores.device)
        mask[valid_token] = 0
        scores_processed = scores.masked_fill(mask, self.filter_value)
        return scores_processed

def get_ocr_processor(
    text_list, tokenizer, 
    model_type, top_k, eos_id
):
    ocr_tree = init_trie(text_list)
    if model_type == 'qwen2vl':
        gen_start_text = 'assistant\n'
    else:
        raise NotImplementedError
    return OCRProcessor(
        ocr_tree, tokenizer, gen_start_text, top_k, eos_id = eos_id
    )