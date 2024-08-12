import os

# --------------------------------------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_type = "7b"
model_family = "llamabase"
result_path = f"/trinity/home/team05/workspace/MIND/auto-labeled/output/{model_family}{model_type}"
# --------------------------------------------- #

from sklearn.metrics import pairwise_distances
print(f"{model_family}{model_type}")

import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer

model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)

import numpy as np
import json
import torch
from tqdm import tqdm
from stats_count import *


def prompt_chat(title):
    return [{"role": "user", "content": f"Question: Tell me something about {title}.\nAnswer: "}]


def get_tokenized_ids(otext, title=None):
    text = otext.replace("@", "").replace("  ", " ").replace("  ", " ")
    text = tokenizer.decode(tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]).replace("<s>", "").replace("</s>", "")
    if model_family == "vicuna":
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Question: Tell me something about {title}.\nAnswer: \nASSISTANT: {text}"
    if "chat" in model_family:
        return chat_change_with_answer(prompt_chat(title), text.strip(), tokenizer)
    return tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()

heads = [
         16,   45,   56,   90,   93,  101,  117,  127,  136,  164,  165,
        205,  214,  223,  227,  231,  244,  250,  251,  260,  262,  267,
        274,  282,  283,  310,  336,  339,  351,  371,  373,  376,  379,
        380,  381,  401,  415,  416,  432,  450,  457,  461,  468,  469,
        477,  495,  501,  506,  511,  514,  549,  589,  607,  622,  626,
        629,  631,  632,  639,  648,  655,  657,  658,  664,  667,  668,
        675,  689,  690,  691,  703,  716,  740,  754,  785,  788,  805,
        818,  833,  837,  854,  862,  863,  865,  895,  898,  901,  918,
        927,  938,  956,  975,  976,  990,  991, 1003, 1009, 1011, 1015
    ]
thresholds_array = [0.1, 0.25]
stats_name = "s_e_v_c_b0b1".split("_")


def get_ats(answer, q):
    q_len = len(get_tokenized_ids('', q))
    ids = get_tokenized_ids(answer, q)
    with torch.no_grad():
        op = model(torch.tensor(ids).to(model.device), output_attentions=True)
    ats = np.array([a.cpu()[0].numpy() for a in op['attentions']])
    ats = ats.reshape(1,32*32,1,ats.shape[-2], ats.shape[-1] )[:,heads]
    stats = count_top_stats_extended(ats, ats[..., :q_len, :q_len], ats[..., q_len:, q_len:], thresholds_array, [], ["q", "s", "c"])
    return stats


# MIND/auto-labeled/output/llamabase7b/data_test.json
for data_type in ["train", "valid", "test"]:
    X, y = [], []
    data = json.load(open(f"/trinity/home/team05/workspace/MIND/auto-labeled/output/llamabase7b/data_{data_type}.json", encoding='utf-8'))

    for j, k in enumerate(tqdm(data)):
        np.save(f"{result_path}/{data_type}_ff/extendedff{j}.0_0.npy", get_ats(k["original_text"], k["title"])) # right
        for i, t in enumerate(k["texts"]):
            np.save(f"{result_path}/{data_type}_ff/extendedff{j}.{i}_1.npy", get_ats(t, k["title"])) # hallu

    
