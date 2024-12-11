import torch
from transformers import AutoTokenizer

from .trie import Trie

class OutputObj(object):
    def __init__(self):
        self.scores = []
        self.sequences = []
        self.past_key_values = None


class OCRAssistModel(object):
    def __init__(self, text_list, model_path, model, model_type):
        self.trie = Trie()
        self.config = model.config
        self.generation_config = model.generation_config
        self.device = model.device
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


        self.JSON_BEGIN = 1
        self.KEY = 2
        self.JSON_KV = 3
        self.VALUE = 4
        self.JSON_VK = 5
        self.JSON_END = 6
        self.status = self.JSON_BEGIN

        self.vocab_size = self.tokenizer.vocab_size
        self.filter_value = -float('inf')

        if self.model_type == 'qwen2_vl':
            # 给出json的特殊字符
            self.json_begin = torch.LongTensor([[515, 220,330]]) # {\n "
            self.json_kv = torch.LongTensor([[788, 330]])      # ": "
            self.json_vk = torch.LongTensor([[756, 220, 330]])   # ",\n  "
            self.json_end = torch.LongTensor([[698, 532]])      # "\n}\n
        self.eos = torch.LongTensor([[151645]])
        self.reset(text_list)
    
    def reset(self, text_list):
        for text in text_list:
            self.trie.insert(text)
        self.status = self.JSON_BEGIN

    def get_output(self, input_ids, out_ids):
        output = OutputObj()
        output.sequences = input_ids
        output.sequences = torch.concat([output.sequences, out_ids.to(output.sequences.device)], dim = 1)
        output.scores = torch.ones(out_ids.shape + (self.vocab_size,)).to(output.sequences.device) * self.filter_value
        for k, id in enumerate(out_ids[0]):
            output.scores[0][k][id] = 1
        
        scores = [score for score in output.scores[0]]
        output.scores = (tuple(scores))
        return output 

    def generate(self, **kwargs):
        if self.status == self.JSON_BEGIN:
            output_ids = self.json_begin
        else:
            output_ids = self.eos

        output = self.get_output(kwargs['input_ids'], output_ids)
        return output

        # input_text = self.tokenizer.decode(kwargs['input_ids'][0])
        # # valid_nodes = self.get_valid_nodes(input_text)
        
        
        # # 需要输出一个object，带scores和sequences
        # # scores 是一个tuple，每个值是(1, 词表大小), 表示词表大小
        # # sequences 是一个tensor表示输出的全部token_id
        # return