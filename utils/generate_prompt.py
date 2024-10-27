B_INST, E_INST = "[INST]", "[/INST]"

summarization_prompt = """Summarize the following news within {word_num} words: {news}"""

helm_prompt = """The following sentence is the first sentence of a Wikipedia article titled {title}. Please continue writing the sentence below. {sentence}"""

def create_ragtruth_prompt(example, source):
    news = source["prompt"]
    word_num = min(200, len(news.split()) // 4)
    prompt = summarization_prompt.format(word_num=word_num, news=news)
    if "INST" in news or "INST" in example['response']:
        raise Exception("has special token")
    res = f"{B_INST} {prompt.strip()} {E_INST} {example['response'].strip()}"
    return res

def create_helm_prompt(example):
    text = example["prompt"]
    title = text.split(".")[0][34:]
    sent = ".".join(text.split(".")[1:])
    prompt = helm_prompt.format(title=title, sentence=sent)
    sentences = ' '.join([x['sentence'] for x in example['sentences']]).strip()
    if "INST" in sent or "INST" in sentences:
        raise Exception("has special token")
    res = f"{B_INST} {prompt.strip()} {E_INST} {sentences}"
    return res