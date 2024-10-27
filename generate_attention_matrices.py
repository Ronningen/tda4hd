import torch
from tqdm import tqdm

from utils.model import get_model
from utils.get_dataset import get_helm_data, get_ragtruth_data
from utils.generate_attentions import get_tokenized_ids, generate_outputs, save_outputs
from utils.generate_prompt import create_ragtruth_prompt, create_helm_prompt

torch.cuda.empty_cache()

helm_pth = "hallu_detection/helm/MIND/helm/data/llamachat7b/data.json"
helm_data = get_helm_data(helm_pth)

model, tokenizer, generation_config, at_id = get_model("7b")
model = model.cuda()
tokenizer.pad_token = "[PAD]"

path = "hallu_detection_ats/helm/"
helm_keys = list(helm_data.keys())
batch_size = 5
for i in tqdm(range(0, len(helm_keys), batch_size)):
    keys = helm_keys[i:i+batch_size]
    prompts = []
    for k in keys:
        prompts.append(create_helm_prompt(helm_data[k])) 
    input_ids, attention_masks, start_inst, end_inst, start_ans, end_ans = get_tokenized_ids(tokenizer, prompts)
    attentions, hidden_state = generate_outputs(model, input_ids, attention_masks)
    torch.cuda.empty_cache()
    save_outputs(path, keys, attentions, input_ids, hidden_state, start_inst, end_inst, start_ans, end_ans)