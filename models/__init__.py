from models.base_model import Local_Model, GPTOSS_Local_Model
from models.r1 import R1_Model
from models.gpt4o import GPT4o_Model
from models.gpt_oss import GPTOSS_Sever_Model

def get_model(model_name, use_vllm=False, gpu_memory_utilization=0.7, max_seq_len=32000, url=None):
    
    # api model
    # if model_name == "gpt-3.5-turbo":
    #     return OpenAI_Model(model_type="gpt-3.5-turbo")
    
    # elif model_name == "gpt-4o":
    #     return OpenAI_Model(model_type="gpt-4o")
    
    # elif model_name == "gpt-4o-mini":
    #     return OpenAI_Model(model_type="gpt-4o-mini")

    # else:
    if model_name == "r1":
        return R1_Model()
    
    if model_name == "gpt-4o":
        return GPT4o_Model()
    
    if "gpt-oss" in model_name or "gpt_oss" in model_name:
        if url is not None:
            return GPTOSS_Sever_Model(model_name, url)
        else:
            return GPTOSS_Local_Model(model_name, max_seq_len=max_seq_len)

    if use_vllm and "baichuan" not in model_name and "gpt_oss" not in model_name:
        from models.vllm_models import VLLM_Model
        return VLLM_Model(model_name, gpu_memory_utilization=gpu_memory_utilization, max_seq_len=max_seq_len)

    else:
        return Local_Model(model_name, max_seq_len=max_seq_len)