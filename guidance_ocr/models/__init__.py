prepare_model_first_inputs_map = {}
update_model_inputs_map = {}
eos_text_map = {}
forward_map = {}


from .qwen2vl import prepare_model_first_inputs as qwen2vl_prepare_first
from .qwen2vl import update_model_inputs as qwen2vl_update

prepare_model_first_inputs_map['qwen2vl'] = qwen2vl_prepare_first
update_model_inputs_map['qwen2vl'] = qwen2vl_update
eos_text_map['qwen2vl'] = ['<|im_end|>']

from .internvl2 import prepare_model_first_inputs as internvl2_prepare_first
from .internvl2 import update_model_inputs as internvl2_update
from .internvl2 import forward as internvl2_forward

prepare_model_first_inputs_map['internvl2'] = internvl2_prepare_first
update_model_inputs_map['internvl2'] = internvl2_update
eos_text_map['internvl2'] = ['<|im_end|>']
forward_map['internvl2'] = internvl2_forward