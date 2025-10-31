
import json
import random
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("../data/webvid_1M.json", "r", encoding="utf-8") as f:
    entries = json.load(f)

random.shuffle(entries)
half = len(entries) // 3
single_prompt_entries = entries[:half]
multi_prompt_entries = entries[half:]

prompt1_template = """You are going to be provided a description about a video, based on the given description, give a question and answer for that video, like you are asking that question about the video.
Your response should be like this:
Question: (your generated question)
Answer: (your generated answer)
So first the question and then the answer.
Only use the information given in the description and not anything else.
Here is the description: 
"""

prompt2_template = """You are going to be provided a description about a video, based on the given description, give a question and answer for that video, like you are asking that question about the video.
Your response should be like this:
Question: (your generated question)
Answer: (your generated answer)
So first the question and then the answer.
Give a multi-turn question answer, like several question answer like a conversation.
Only use the information given in the description and not anything else.
Here is the description: 
"""

def generate_response_batch(prompts):
    messages_list = [
        [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": p}
        ]
        for p in prompts
    ]
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)


    
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    res = []
    for r in responses:
        rr = r.strip().split('assistant\n')
        if len(rr) > 1:
            res.append(rr[1])
        else:
            res.append(rr[0])
    
    return res

def parse_single_qa(response):
    lines = response.splitlines()
    q = a = ""
    for line in lines:
        if line.lower().startswith("question:"):
            q = line[len("question:"):].strip()
        elif line.lower().startswith("answer:"):
            a = line[len("answer:"):].strip()
    return {"question": q, "answer": a, "history": []}

def parse_multi_qa(response):
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    qa_pairs = []
    for i in range(0, len(lines) - 1):
        if lines[i].lower().startswith("question:") and lines[i+1].lower().startswith("answer:"):
            q = lines[i][len("question:"):].strip()
            a = lines[i+1][len("answer:"):].strip()
            qa_pairs.append((q, a))
    if not qa_pairs:
        return None
    *history, (last_q, last_a) = qa_pairs
    return {"question": last_q, "answer": last_a, "history": history}

def process_in_batches(entries, template, parser, batch_size=300):
    dataset = []
    for i in tqdm(range(0, len(entries), batch_size)):
        batch = entries[i:i+batch_size]
        prompts = [template + item["name"] for item in batch]
        responses = generate_response_batch(prompts)
        # print(responses)
        for item, response in zip(batch, responses):
            result = parser(response)
            if result and result["question"] and result["answer"]:
                dataset.append({
                    "videoid": item["videoid"],
                    "question": result["question"],
                    "answer": result["answer"],
                    "history": result["history"]
                })
    return dataset

print("Generating single-turn QA samples...")
output_dataset = process_in_batches(single_prompt_entries, prompt1_template, parse_single_qa)

print("Generating multi-turn QA samples...")
output_dataset += process_in_batches(multi_prompt_entries, prompt2_template, parse_multi_qa)

with open("subsets/webvid_qa.json", "w", encoding="utf-8") as f:
    json.dump(output_dataset, f, indent=2, ensure_ascii=False)

print("Done")

