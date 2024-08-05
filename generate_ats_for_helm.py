import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer, get_pe
import json


from tqdm import tqdm

model_type = "7b"
model_family = "llamabase"
print(f"{model_family}{model_type}")

def prompt_chat_for_labeled(prompt):
    return [{"role": "user", "content": prompt}]


def get_tokenized_ids(answer, q):

    text = f"{q.strip()} {answer.strip()}"
    otext = q.strip()

    if "chat" in model_family:
        id1 = chat_change_with_answer(prompt_chat_for_labeled(q), answer.strip(), tokenizer)[0]
        start_at = -1
        for i in range(len(id1)):
            if id1[i:i+4] == [518, 29914, 25580, 29962]:
                start_at = i
        if start_at == -1:
            raise Exception
        else:
            start_at += 4

    else:
        id1 = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0]
        id2 = tokenizer(otext.strip(), return_tensors='pt')['input_ids'].tolist()[0]
        start_at = -1
        for i in range(len(id1)):
            if i >= len(id2) or id1[i] != id2[i]:
                start_at = i
                break
    return [id1], start_at

def get_ats(answer, q):
    ids, _ = get_tokenized_ids(answer, q)
    with torch.no_grad():
        op = model(torch.tensor(ids).to(model.device), output_attentions=True)
    ats = [a.cpu()[0].tolist() for a in op['attentions']]
    return ats


model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)


ats_result_path = f"./helm/ats/{model_family}{model_type}"


if not os.path.exists(ats_result_path):
    os.mkdir(ats_result_path)
with open(f"./helm/data/{model_family}{model_type}/data.json", "r", encoding='utf-8') as f:
    data = json.load(f)
result = {}
for d in tqdm(data):
    result[d] = {
        "sentences": [],
        "passage": None,
    }
    prompt = data[d]["prompt"]
    prompt = tokenizer.decode(tokenizer(prompt.strip(), return_tensors='pt')['input_ids'].tolist()[0]).replace("<s>", "").replace("</s>", "")
    for s in data[d]["sentences"]:
        ats = get_ats(s["sentence"], prompt)
        result[d]["sentences"].append(ats)
    ats = get_ats(" ".join([t["sentence"] for t in data[d]["sentences"]]), prompt)
    result[d]["passage"] = ats
with open(f"{ats_result_path}/ats.json", "w+") as f:
    json.dump(result, f)
