if __name__ == "__main__":

    
   

    from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
    import torch
    import av
    import numpy as np

    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)

    processor = LlavaNextVideoProcessor.from_pretrained(model_id)

    def read_video_pyav(container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

            

    output_file = 'output.txt'

    def append_output(label, response):
        with open(output_file, 'a') as f:
            f.write(f"Label: {label}\n")
            f.write(f"Response: {response}\n")
            f.write("-" * 40 + "\n")  


    import json

    input_json_file = "subsets/test.json"  

  
    with open(input_json_file, "r") as file:
        data = json.load(file)


    for sample in data:
        messages = sample["messages"]
        transformed_messages = []
        

        content = messages[0]["content"]
        
        parts = content.split("<video>")
        transformed_content = []

        for i, part in enumerate(parts):
            
            if part.strip():
                transformed_content.append({"type": "text", "text": part.strip()})
            
            if i < len(parts) - 1:
                transformed_content.append({"type": "video"})
        
       
        transformed_messages.append({
            "role": "user",
            "content": transformed_content
        })

        prompt = processor.apply_chat_template(transformed_messages, add_generation_prompt=True)

        input_videos = []

        for vid in sample["videos"]:
            container = av.open(vid)

            # sample uniformly 8 frames from the video, can sample more for longer videos
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip = read_video_pyav(container, indices)
            input_videos.append(clip)
        
        
        input_model = processor(text=prompt, videos=input_videos, padding=True, return_tensors="pt").to(model.device)

        output = model.generate(**input_model, max_new_tokens=8182, do_sample=False)
        
        # Append the transformed conversation
        append_output(sample['messages'][1]["content"], processor.decode(output[0][2:], skip_special_tokens=True))
        
