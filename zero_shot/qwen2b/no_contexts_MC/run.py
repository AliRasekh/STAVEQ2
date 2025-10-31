if __name__ == "__main__":

    
   

    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch
    import random
    import os

    model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32, device_map="auto"
    )
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float32, min_pixels=min_pixels, max_pixels=max_pixels)

            

    output_file = 'output.txt'

    def append_output(label, response):
        with open(output_file, 'a') as f:
            f.write(f"Label: {label}\n")
            f.write(f"Response: {response}\n")
            f.write("-" * 40 + "\n")  


    import json

    # File paths
    input_json_file = "subsets/test.json"  

    # Load the JSON data
    with open(input_json_file, "r") as file:
        data = json.load(file)


    for sample in data:
        text = processor.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(sample)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        append_output(sample['messages'][1]["content"], output_text[0])
        
