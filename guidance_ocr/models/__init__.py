from .qwen2vl import prepare_model_first_inputs as qwen2vl_prepare_first
from .qwen2vl import update_model_inputs as qwen2vl_update

from .llms import prepare_model_first_inputs as llms_prepare_first
from .llms import update_model_inputs as llms_update

prepare_model_first_inputs_map = {}
prepare_model_first_inputs_map['qwen2vl'] = qwen2vl_prepare_first
prepare_model_first_inputs_map['llms'] = llms_prepare_first

update_model_inputs_map = {}
update_model_inputs_map['qwen2vl'] = qwen2vl_update
update_model_inputs_map['llms'] = llms_update

eos_text_map = {}
eos_text_map['qwen2vl'] = ['<|im_end|>', '<|endoftext|>']
eos_text_map['llms'] = ['<|im_end|>', '<|endoftext|>']