import re
import torch
import random
from transformers import AutoTokenizer

from .trie import Trie

class OutputObj(object):
    def __init__(self):
        self.scores = []
        self.sequences = []
        self.past_key_values = None

def init_trie(text_list):
    trie = Trie()
    for text in text_list:
        trie.insert(text)
    return trie

def qwen2vl_text2id(tokenizer, text):
    return torch.LongTensor([tokenizer(text)['input_ids']])

def get_valid_text_from_root(generated_text, root_node):
    if root_node.char is None:
        return ''
        
    if generated_text != '':
        for node_key in root_node.children.keys(): # 字典树中一个node不可能有两个相同char的子节点
            if node_key == generated_text[0]:
                return get_valid_text_from_root(generated_text[1:], root_node.children[node_key])
        return ''
    else:
        if len(root_node.children) == 1:
            for key in root_node.children.keys():
                return root_node.char + get_valid_text_from_root('', root_node.children[key])
        else: # 为0或着出现多个，均需要结束掉
            return root_node.char


def get_valid_texts(generated_text, all_nodes):
    cand_text_list = []

    for node in all_nodes:
        if node.char is not None and node.char == generated_text[0]:
            cand_text = get_valid_text_from_root(
                generated_text[1:], node
            )

            if len(cand_text) > 1:
                cand_text_list.append(cand_text[1:])
    return cand_text_list

class JsonAssistModel(object):
    def __init__(self, text_list, extract_keys, model_path, model, model_type):
        self.trie = init_trie(text_list)
        self.ocr_all_nodes = self.trie.get_init_avinodes(self.trie.root)

        self.extract_keys = extract_keys
        self.config = model.config
        self.generation_config = model.generation_config
        self.device = model.device
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


        self.JSON_START = 0
        self.KEY = 1
        self.JSON_KV = 2
        self.VAL = 3
        self.JSON_VKEND = 4
        self.filter_value = -float('inf')

        if self.model_type == 'qwen2vl':
            # 给出json的特殊字符
            self.json_begin = torch.LongTensor([[515, 220,330]]) # {\n "
            self.json_kv = torch.LongTensor([[788, 330]])      # ": "
            self.json_vk = torch.LongTensor([[756, 220, 330]])   # ",\n  "
            self.json_end = torch.LongTensor([[698, 532]])      # "\n}
            self.eos = torch.LongTensor([[151645]])

            # 给出模型用于生成的字符串
            self.gen_start_text = 'assistant\n'
            self.text2id = qwen2vl_text2id

            self.logits_shape = 151936

       

    def get_output(self, input_ids, out_ids):
        output = OutputObj()
        output.sequences = input_ids
        output.sequences = torch.concat([output.sequences, out_ids.to(output.sequences.device)], dim = 1)
        output.scores = torch.ones(out_ids.shape + (self.logits_shape,)).to(output.sequences.device) * self.filter_value
        
        try:
            for k, id in enumerate(out_ids[0]):
                output.scores[0][k][id] = 1
        except:
            import pdb; pdb.set_trace()
        
        scores = [score for score in output.scores[0]]
        output.scores = (tuple(scores))
        return output 

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


    def generate(self, **kwargs):
        input_text = self.tokenizer.decode(kwargs['input_ids'][0])
        generated_text = input_text.split(self.gen_start_text)[-1]
        self.state = self.get_cur_state(generated_text)
        if self.JSON_START in self.state:
            output_ids = self.json_begin
        elif self.KEY in self.state: # 包含self.Key的状态为[self.KEY, self.JSON_KV]
            key_text = generated_text.split('\"')[-1]
            if len(key_text) == 0:
                output_ids = self.json_kv
            
            cand_key_list = []
            for cand_key in self.extract_keys:
                if len(cand_key) >= len(key_text) and \
                    key_text == cand_key[:len(key_text)]:
                    cand_key_list.append(cand_key[len(key_text):])
            if len(cand_key_list) > 0:
                output_ids = self.text2id(self.tokenizer, random.choice(cand_key_list))
                output_ids = torch.cat((output_ids, self.json_kv), dim = 1)
            else:
                output_ids = self.json_kv
        elif self.VAL in self.state: # 包含self.VAL的状态为[self.VAL, self.JSON_VKEND]
            val_text = generated_text.split('\"')[-1]
            if len(val_text) == 0:
                output_ids = self.json_vk

            cand_text_list = get_valid_texts(val_text, self.ocr_all_nodes)
            if len(cand_text_list) > 0:
                output_ids = self.text2id(self.tokenizer, random.choice(cand_text_list))
                output_ids = torch.cat((output_ids, self.json_vk), dim = 1)
            else:
                output_ids = self.json_vk
        else:
            output_ids = self.eos


        output = self.get_output(kwargs['input_ids'], output_ids)
        return output


class OCRAssistModel(object):
    def __init__(self, text_list, model_path, model, model_type):
        self.trie = init_trie(text_list)
        self.ocr_all_nodes = self.trie.get_init_avinodes(self.trie.root)
        self.config = model.config
        self.generation_config = model.generation_config
        self.device = model.device
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.filter_value = -float('inf')

        if self.model_type == 'qwen2vl':
            self.eos = torch.LongTensor([[151645]])

            # 给出模型用于生成的字符串
            self.gen_start_text = 'assistant\n'
            self.text2id = qwen2vl_text2id

            self.logits_shape = 151936
        else:
            raise NotImplementedError  

    def get_output(self, input_ids, out_ids):
        output = OutputObj()
        output.sequences = input_ids
        output.sequences = torch.concat([output.sequences, out_ids.to(output.sequences.device)], dim = 1)
        output.scores = torch.ones(out_ids.shape + (self.logits_shape,)).to(output.sequences.device) * self.filter_value
        
        try:
            for k, id in enumerate(out_ids[0]):
                output.scores[0][k][id] = 1
        except:
            import pdb; pdb.set_trace()
        
        scores = [score for score in output.scores[0]]
        output.scores = (tuple(scores))
        return output 

    def generate(self, **kwargs):
        input_text = self.tokenizer.decode(kwargs['input_ids'][0])
        generated_text = input_text.split(self.gen_start_text)[-1]
        
        val_text = generated_text.split('\"')[-1]
        if len(val_text) == 0:
            output_ids = self.eos
        else:
            cand_text_list = get_valid_texts(val_text, self.ocr_all_nodes)
            if len(cand_text_list) > 0:
                output_ids = self.text2id(self.tokenizer, random.choice(cand_text_list))
            else:
                output_ids = self.eos


        output = self.get_output(kwargs['input_ids'], output_ids)
        return output