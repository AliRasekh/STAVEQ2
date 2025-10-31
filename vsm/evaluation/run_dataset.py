import torch
from transformers import AutoModel, Qwen2VLForConditionalGeneration
import os, json, sys
from dataset import SSv2

train_backbone_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


sys.path.append(train_backbone_path)

from peft import PeftModel
from custom_model import CustomQwen2VLForConditionalGeneration


annotations_path = '../subsets/test.json'

dataset = SSv2()



model = Qwen2VLForConditionalGeneration.from_pretrained(
    '../saves/base', torch_dtype="auto", device_map="auto"
).to('cuda')




with open(annotations_path, 'r') as f:
    annotations = json.loads(f.read())



output_file = 'base.txt'

def append_output(label, response):
    with open(output_file, 'a') as f:
        f.write(f"Label: {label}\n")
        f.write(f"Response: {response}\n")
        f.write("-" * 40 + "\n")  

for i in range(len(annotations)):
    annotation = annotations[i]
    label = annotation[1]['content'][0]['text']
    inputs = dataset.collate([annotation]).to('cuda')
    

    generated_ids = model.generate(**inputs, max_new_tokens=1024)

    output_text = dataset.preprocessor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    append_output(label ,output_text[0])



    torch.cuda.empty_cache()
