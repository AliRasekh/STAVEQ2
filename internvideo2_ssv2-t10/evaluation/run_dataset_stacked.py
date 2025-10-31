import torch
import os, json, sys
from dataset import SSv2
from transformers import AutoProcessor
from peft import PeftModel

backbone_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


sys.path.append(backbone_path)

from modeling_videochat2 import InternVideo2_VideoChat2

annotations_path = '../subsets/100_percent_test.json'

dataset = SSv2()

model = InternVideo2_VideoChat2.from_pretrained(
    '../base_model', torch_dtype=torch.bfloat16
).to('cuda')


processor = AutoProcessor.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)



with open(annotations_path, 'r') as f:
    annotations = json.loads(f.read())

finetuned_model = PeftModel.from_pretrained(model,
                                  '../saves/internvideo2_base/lora',
                                  torch_dtype=torch.float16,
                                  is_trainable=False,
                                  device_map="auto"
                                  )


output_file = 'test_base.txt'

def append_output(label, response):
    with open(output_file, 'a') as f:
        f.write(f"Label: {label}\n")
        f.write(f"Response: {response}\n")
        f.write("-" * 40 + "\n")  

for i in range(len(annotations)):
    annotation = annotations[i]
    label = annotation[1]['content'][0]['text']
    inputs = dataset.collate([annotation])
    

    generated_ids = finetuned_model.generate_caption(**inputs, max_new_tokens=1024)

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    append_output(label ,output_text[0])



    torch.cuda.empty_cache()
