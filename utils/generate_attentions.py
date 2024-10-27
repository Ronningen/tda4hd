import json
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import set_seed

np.random.seed(23)
torch.manual_seed(23)
set_seed(23)

def generate_for_counterfact(cf_df, gen_config, model, tokenizer, output_path):
    idx2gen = {}
    for _, row in tqdm(cf_df.iterrows()):
        # generate prompt
        idx = row["case_id"]
        rewrite = row["requested_rewrite"]
        prompt = rewrite["prompt"]
        subj = rewrite["subject"]
        target = rewrite["target_true"]["str"]
        prompt = prompt.format(subj)
        if len(prompt.split()) < len(subj.split()) + 3:
            prompt = row["generation_prompts"][0]
        idx2gen[idx] = {"prompt": prompt, "target": target}
        # generate answer
        with torch.no_grad():
            out = tokenizer([prompt], padding=True)
            input_ids = out['input_ids']
            attention_masks = out['attention_mask']
            gen = model.generate(inputs=torch.tensor(input_ids).to(model.device), generation_config=gen_config)
        gen_text = tokenizer.decode(gen.sequences[0])
        idx2gen[idx]["generated"] = gen_text
        # save everything
        attentions_tmp = np.stack([x.cpu() for x in gen["attentions"][-1]]).squeeze()
        np.save(output_path + str(idx) + "_attentions.npy", attentions_tmp)
        hidden_state_tmp = gen["hidden_states"][-1][-1].cpu().squeeze()
        np.save(output_path + str(idx) + "_hidden_states.npy", hidden_state_tmp)
        token_ids_tmp = np.array(input_ids[0])
        np.save(output_path + str(idx) + "_input_token_ids.npy", token_ids_tmp)
        token_ids_tmp = np.array(gen.sequences[0].cpu())
        np.save(output_path + str(idx) + "_output_token_ids.npy", token_ids_tmp)
    with open(output_path + "counterfact_generated.json", "w") as f:
        json.dump(idx2gen, f)

def get_tokenized_ids(tokenizer, texts):
    """
    Generates tokenized ids for a list of texts
    Args:
        tokenizer: Tokenizer
        texts: list of texts, containing instructions and generated texts
    Returns:
        input_ids
        attention_masks
        lists of indices for the start and end of each instruction and generated texts
    """
    out = tokenizer(texts, padding=True)
    input_ids = out['input_ids']
    attention_masks = out['attention_mask']
    start_inst = [4 for _ in range(len(input_ids))]
    end_inst = []
    start_ans = []
    end_ans = []
    for i, ids in enumerate(input_ids):
        # find index of the last [/INST] token
        idx = len(ids) - ids[::-1].index(25580)
        end_inst.append(idx - 3)
        start_ans.append(idx + 1)
        if 0 in attention_masks[i]:
            end_ans.append(attention_masks[i].index(0))
        else:
            end_ans.append(len(input_ids[i]))
    return input_ids, attention_masks, start_inst, end_inst, start_ans, end_ans

def generate_outputs(model, input_ids, attention_masks):
    with torch.no_grad():
        op = model.forward(torch.tensor(input_ids).to(model.device), torch.tensor(attention_masks).to(model.device), output_attentions=True, output_hidden_states=True)
    return np.array(torch.stack(op.attentions).cpu()), np.array(op.hidden_states[-1].cpu())

def save_outputs(path, prefixes, attentions, token_ids, hidden_state, start_inst, end_inst, start_ans, end_ans):
    # save attentions, truncated
    bs = attentions.shape[1]
    for i in range(bs):
        attentions_tmp = attentions[:, i, :, start_inst[i]:end_ans[i], start_inst[i]:end_ans[i]]
        np.save(path + prefixes[i] + "_attentions.npy", attentions_tmp)
        hidden_state_tmp = hidden_state[i, start_inst[i]:end_ans[i], :]
        np.save(path + prefixes[i] + "_hidden_states.npy", hidden_state_tmp)
        token_ids_tmp = np.array(token_ids[i][start_inst[i]:end_ans[i]])
        np.save(path + prefixes[i] + "_token_ids.npy", token_ids_tmp)
        params = np.array([start_inst[i], end_inst[i], start_ans[i], end_ans[i]])
        np.save(path + prefixes[i] + "_params.npy", params)