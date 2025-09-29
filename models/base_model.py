import os
import re
import pdb
import yaml
import torch
try:
    import torch_npu
except:
    pass
from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, AutoConfig
from accelerate import PartialState
device_type = PartialState().default_device.type

current_dir = os.path.dirname(os.path.abspath(__file__))

##

with open(os.path.join(current_dir, 'config.yaml'), 'r') as f:
    # 将YAML内容转换为字典
    LOCAL_MODEL_PATHS = yaml.safe_load(f)

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

class Base_Model:
    def __init__(self, model_name):
        self.model_path = self.get_model_path(model_name)
        if os.path.exists(os.path.join(self.model_path, "adapter_model.safetensors")):
            self.peft_path = self.model_path
            lora_config = LoraConfig.from_pretrained(self.peft_path)
            self.model_path = lora_config.base_model_name_or_path
        else:
            self.peft_path = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def __call__(self):
        raise NotImplementedError
    
    def prepare_inputs(self, inputs, enable_thinking=False):
        if isinstance(inputs, str) or isinstance(inputs[0], str):
            return inputs
        
        elif isinstance(inputs, list) or isinstance(inputs[0], list):
            if "qwen3" in self.model_path.lower():
                if inputs[-1][-1]["role"] == "user":
                    return self.tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
                else:
                    input_text = self.tokenizer.apply_chat_template(inputs, add_generation_prompt=False, tokenize=False, enable_thinking=enable_thinking)
                    input_text = [input_.rsplit("<|im_end|>", 1)[0] for input_ in input_text]
                    # input_text[-1] = input_text[-1].rsplit("<|im_end|>", 1)[0]
                    return input_text
            elif "baichuan-m2" in self.model_path.lower():
                if inputs[-1][-1]["role"] == "user":
                    return self.tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False, thinking_mode="on" if enable_thinking else "off")
                else:
                    input_text = self.tokenizer.apply_chat_template(inputs, add_generation_prompt=False, tokenize=False, thinking_mode="on" if enable_thinking else "off")
                    input_text = [input_.rsplit("<|im_end|>", 1)[0] for input_ in input_text]
                    # input_text[-1] = input_text[-1].rsplit("<|im_end|>", 1)[0]
                    return input_text
            elif "ehr-r1" in self.model_path.lower():
                assert inputs[-1][-1]["role"] == "user"
                input_text = self.tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
                new_input_text = []
                if enable_thinking is False:
                    for input_ in input_text:
                        if not input_.endswith("</think>"):
                            input_ = input_ + "<think>\n\n</think>\n"
                        new_input_text.append(input_)

                return new_input_text

            else:
                if inputs[-1][-1]["role"] == "user":
                    return self.tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False)
                else:
                    input_text = self.tokenizer.apply_chat_template(inputs, add_generation_prompt=False, tokenize=False)
                    input_text = [input_.rsplit("<|im_end|>", 1)[0] for input_ in input_text]
                    # input_text[-1] = input_text[-1].rsplit("<|im_end|>", 1)[0]
                    return input_text
                # assert inputs[-1][-1]["role"] == "user", "the final messages of the propmt must be user!"
                # return self.tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False)

        else:
            raise NotImplementedError(f"""The inputs type {type(inputs)} not in (str, list[str], dict, list[dict])""")
    
    def get_logit_bias(self, word_list=[]):
        logit_bias = {}
        # pdb.set_trace()
        for word in word_list:
            for token_id in self.tokenizer(word, add_special_tokens=False)["input_ids"]:
                logit_bias[token_id] = 100

        return logit_bias
    
    def get_model_path(self, model_name):
        return model_name if model_name not in LOCAL_MODEL_PATHS else LOCAL_MODEL_PATHS[model_name]

class Local_Model(Base_Model):
    def __init__(self, model_name, max_seq_len=16000):
        super().__init__(model_name)

        disable_torch_init()
        # self.device = "npu:0" if torch.npu.is_available() else "cuda:0"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
            torch_dtype=torch.float16 if not "gpt-oss" in self.model_path else torch.bfloat16, trust_remote_code=True, device_map="auto")

        self.device = self.model.device    
        self.tokenizer.padding_side = "left" 
        
        model_config = AutoConfig.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
        self.max_seq_len = min(getattr(model_config, "max_position_embeddings", max_seq_len), max_seq_len)
        
        if self.peft_path is not None:
            lora_config = LoraConfig.from_pretrained(self.peft_path)
            self.model = PeftModel.from_pretrained(self.model, self.peft_path, config=lora_config)
            print(f"Merging weights")
            self.model = self.model.merge_and_unload()
            print('Convert to FP16...')
            self.model.to(torch.float16)
        self.model.eval()
    
    def __call__(self, inputs, infer_args, logit_bias_words=False, enable_thinking=False):
        inputs = self.prepare_inputs(inputs, enable_thinking)
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key][:, -self.max_seq_len+infer_args["max_tokens"]:]
        
        if logit_bias_words:
            logit_bias = self.get_logit_bias(logit_bias_words)
            logits_processor_list = LogitsProcessorList([
                LogitBiasLogitsProcessor(logit_bias),
            ])

        with torch.no_grad():
            model_outputs = self.model.generate(
                input_ids=inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                num_return_sequences=infer_args["n"],
                do_sample=False if infer_args["temperature"] == 0.0 else True,
                temperature=infer_args["temperature"],
                top_k=infer_args["top_k"],
                top_p=infer_args["top_p"],
                max_new_tokens=infer_args["max_tokens"],
                tokenizer=self.tokenizer,
                logits_processor=logits_processor_list if logit_bias_words else None,
                return_dict_in_generate=True, 
                output_logits=True
            )
            
        final_model_outputs = model_outputs.sequences[:, len(inputs["input_ids"][0]):]

        if logit_bias_words:
            logit_bias_id = torch.tensor(list(logit_bias.keys())).to(model_outputs.sequences)
            logits = model_outputs.logits[0][:, logit_bias_id].softmax(-1).max(-1)[0]
        else:
            logits = None

        outputs = self.tokenizer.batch_decode(final_model_outputs, skip_special_tokens=True)
        # outputs = [outputs[batch_id*sample_num:(batch_id+1)*sample_num] for batch_id in range(len(inputs.input_ids))]

        final_outputs = []
        for batch_id in range(len(inputs.input_ids)):
            batch_outputs = []
            for i in range(infer_args["n"]):
                batch_outputs.append(
                    {
                        "trajectory": outputs[batch_id*infer_args["n"]+i],
                        "logits": logits[batch_id*infer_args["n"]+i].item() if logits else None,
                    }
                )
            final_outputs.append(batch_outputs)
        

        return final_outputs

class GPTOSS_Local_Model(Local_Model):

    def __call__(self, inputs, infer_args, logit_bias_words=False, enable_thinking=False):
        inputs = self.prepare_inputs(inputs, enable_thinking)
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key][:, -self.max_seq_len+infer_args["max_tokens"]:]
        
        if logit_bias_words:
            logit_bias = self.get_logit_bias(logit_bias_words)
            logits_processor_list = LogitsProcessorList([
                LogitBiasLogitsProcessor(logit_bias),
            ])

        with torch.no_grad():
            model_outputs = self.model.generate(
                input_ids=inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                num_return_sequences=infer_args["n"],
                do_sample=False if infer_args["temperature"] == 0.0 else True,
                temperature=infer_args["temperature"],
                top_k=infer_args["top_k"],
                top_p=infer_args["top_p"],
                max_new_tokens=infer_args["max_tokens"],
                tokenizer=self.tokenizer,
                logits_processor=logits_processor_list if logit_bias_words else None,
                return_dict_in_generate=True, 
                output_logits=True
            )
            
        final_model_outputs = model_outputs.sequences[:, len(inputs["input_ids"][0]):]

        if logit_bias_words:
            logit_bias_id = torch.tensor(list(logit_bias.keys())).to(model_outputs.sequences)
            logits = model_outputs.logits[0][:, logit_bias_id].softmax(-1).max(-1)[0]
        else:
            logits = None

        outputs = self.tokenizer.batch_decode(final_model_outputs, skip_special_tokens=False)
        # outputs = [outputs[batch_id*sample_num:(batch_id+1)*sample_num] for batch_id in range(len(inputs.input_ids))]

        final_outputs = []
        for batch_id in range(len(inputs.input_ids)):
            batch_outputs = []
            for i in range(infer_args["n"]):
                batch_outputs.append(
                    {
                        "trajectory": outputs[batch_id*infer_args["n"]+i],
                        "logits": logits[batch_id*infer_args["n"]+i].item() if logits else None,
                    }
                )
            final_outputs.append(batch_outputs)

        return final_outputs

class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, logit_bias):
        self.logit_bias = logit_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

        for index in self.logit_bias.keys():
            scores[:, index] += self.logit_bias[index]
        return scores
