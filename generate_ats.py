import os

# --------------------------------------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
model_type = "7b"
model_family = "llamabase"
result_path = f"./auto-labeled/output/{model_family}{model_type}"
# --------------------------------------------- #


print(f"{model_family}{model_type}")

import torch
from utils.model import get_model
from utils.gen import chat_change_with_answer

model, tokenizer, generation_config, at_id = get_model(model_type, model_family, 1)




from tqdm import tqdm
import json

import json
import torch
from tqdm import tqdm



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

def get_ats(answer, q):
    ids, _ = get_tokenized_ids(answer, q)
    with torch.no_grad():
        op = model(torch.tensor(ids).to(model.device), output_attentions=True)
    ats = [a.cpu()[0].tolist() for a in op['attentions']]
    return ats


for data_type in ["train", "valid", "test"]:
    data = json.load(open(f"{result_path}/data_{data_type}.json", encoding='utf-8'))
    results = []

    for k in tqdm(data):
        results.append({
            "right": get_ats(k["original_text"], k["title"]),
            "hallu": [get_ats(t, k["title"]) for  t in k["texts"]],
        })

    with open(f"{result_path}/ats_{data_type}.json", "w+") as f:
        json.dump(results, f)


