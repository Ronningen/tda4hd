import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig
from transformers import set_seed

from utils.model import get_model
from utils.get_dataset import get_helm_data, get_ragtruth_data
from utils.generate_attentions import generate_for_counterfact
from utils.generate_prompt import create_ragtruth_prompt, create_helm_prompt

np.random.seed(23)
torch.manual_seed(23)
set_seed(23)

model, tokenizer, generation_config, at_id = get_model("7b", dtype=torch.float32)
# model = model.cuda()
tokenizer.pad_token = "[PAD]"
cf_path = "hallu_detection/counterfact_train.parquet"
df = pd.read_parquet(cf_path)
path = "hallu_detection/hallu_detection_ats/counterfact/"
gen_conf = GenerationConfig(
    max_new_tokens=40,
    use_cache=False,
    temperature=1.0,
    output_attentions=True,
    output_hidden_states=True,
    return_dict_in_generate=True,
    bos_token_id=1,
    eos_token_id=2)
generate_for_counterfact(df.iloc[:3000], gen_conf, model, tokenizer, path)