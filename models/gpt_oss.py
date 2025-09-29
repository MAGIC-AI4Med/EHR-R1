import requests
import time
import math

from models.base_model import Base_Model


def get_logits_sglang(content, logit_bias_words):
    for token_info in content[::-1]:
        token_string = token_info["token"]
        if token_string not in logit_bias_words:
            continue
        
        logit = token_info["logprob"]
        break
    
    return logit


class GPTOSS_Sever_Model(Base_Model):
    def __init__(self, model_name, url):
        super().__init__(model_name)
        assert url is not None
        self.url = f"{url}/v1/chat/completions"
        self.max_try = 5

    def __call__(self, inputs, infer_args, logit_bias_words=None, enable_thinking=False):
        # inputs = self.prepare_inputs(inputs, enable_thinking)
        data = {
            "model": self.model_path,
            "messages": inputs[0],
            "logit_bias_words": self.get_logit_bias(logit_bias_words) if logit_bias_words else None,
            "max_tokens": 4096,
            "temperature": infer_args["temperature"],
            "n": infer_args["n"],
            "top_k": infer_args["top_k"],
            "top_p": infer_args["top_p"],
            "logprobs": True
        }

        try_num = 0
        while try_num<self.max_try:
            response = requests.post(self.url, json=data).json()
            try:
                outputs = [
                    [
                        {
                            "reasoning": output["message"]["reasoning_content"].strip() if output["message"]["reasoning_content"] is not None else output["message"]["reasoning_content"],
                            "trajectory": output["message"]["content"].strip(),
                            "logits": math.exp(get_logits_sglang(output["logprobs"]["content"], logit_bias_words)) if logit_bias_words is not None else None#[ for token_info in output["logprobs"]["content"]],
                        }
                    ] for output in response["choices"]
                ]
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error when requesting models, Retry after 60 seconds...")
                time.sleep(60)
                print(f"Retry[{try_num}/{self.max_try}]...")
                try_num += 1
            
        return outputs
            # except:
            #     print(f"Error when requesting models, Retry after 60 seconds...")
            #     time.sleep(60)
            #     print(f"Retry[{try_num}/{self.max_try}]...")
            #     try_num += 1
            #     continue
