import os
import torch
import math
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from models.base_model import Base_Model
from transformers import AutoConfig

# os.environ["VLLM_INSTALL_PUNICA_KERNELS"] = 1

class VLLM_Model(Base_Model):
    def __init__(self, model_name, gpu_memory_utilization=0.7, max_seq_len=32000):
        super().__init__(model_name)
        model_config = AutoConfig.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
        self.max_seq_len = min(getattr(model_config, "max_position_embeddings", max_seq_len), max_seq_len)

        print(f"Loading model on {torch.cuda.device_count()} devices...")
        if self.peft_path is None:
            self.model = LLM(model=self.model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, max_model_len=self.max_seq_len, max_seq_len_to_capture=self.max_seq_len, gpu_memory_utilization=gpu_memory_utilization) # gpu_memory_utilization=0.8
        else:
            self.model = LLM(model=self.model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, max_model_len=self.max_seq_len, max_seq_len_to_capture=self.max_seq_len, gpu_memory_utilization=gpu_memory_utilization, enable_lora=True) # gpu_memory_utilization=0.8
    
    def model_forward(self, inputs, sampling_params):
        input_ids = self.tokenizer(inputs)["input_ids"]
        input_ids = [input_id[-self.max_seq_len+sampling_params.max_tokens:] for input_id in input_ids]
        if self.peft_path:
            outputs = self.model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params, lora_request=LoRARequest("lora", 1, self.peft_path), use_tqdm=False)
        else:
            outputs = self.model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params, use_tqdm=False)
        return outputs

    def __call__(self, inputs, infer_args, logit_bias_words=None, enable_thinking=False):
        inputs = self.prepare_inputs(inputs, enable_thinking=enable_thinking)
        sampling_params = SamplingParams(logit_bias=None if not logit_bias_words else self.get_logit_bias(logit_bias_words), logprobs=1, **infer_args)
        outputs = self.model_forward(inputs, sampling_params)

        outputs = [
            [
                {
                    "trajectory": output.text.strip(),
                    "logits": math.exp(output.cumulative_logprob) if logit_bias_words else math.exp(sum([list(logprob.values())[0].logprob for logprob in output.logprobs[:-1]]))
                }
                for output in outputs[i].outputs
            ] 
            for i in range(len(inputs))
        ]


        return outputs
