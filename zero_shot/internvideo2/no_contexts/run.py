if __name__ == "__main__":

    
    IMG_TOKEN = "[<IMG_PLH>]"
    VID_TOKEN = "[<VID_PLH>]"

    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_BOS_TOKEN = '<s>'
    DEFAULT_EOS_TOKEN = '</s>'
    DEFAULT_UNK_TOKEN = "<unk>"

    DEFAULT_IMAGE_TOKEN = "[IMAGETOKEN]"
    DEFAULT_VIDEO_TOKEN = "[VIDEOTOKEN]"

    DEFAULT_IMG_PLACEHOLDER = "[<IMG_PLH>]"
    DEFAULT_VID_PLACEHOLDER = "[<VID_PLH>]"

    import os
    token = ''
    import torch


    from transformers import AutoTokenizer, AutoModel
    tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)

    model = AutoModel.from_pretrained(
        'OpenGVLab/InternVideo2-Chat-8B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True).cuda()

    from decord import VideoReader, cpu
    from PIL import Image
    import numpy as np
    import numpy as np
    import decord
    from decord import VideoReader, cpu
    import torch.nn.functional as F
    import torchvision.transforms as T
    from torchvision.transforms import PILToTensor
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    decord.bridge.set_bridge("torch")

        
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets


    def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std)
        ])

        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        frames = transform(frames)

        T_, C, H, W = frames.shape
            
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return frames, msg
        else:
            return frames


    def chat_multiple_media(
        model,
        tokenizer,
        conversation,
        msg,
        user_prompt,
        media_type,
        media_tensors, 
        instruction=None,
        chat_history = [],
        return_history = False,
        generation_config = {}
        ):
            """
            A function to process multiple media inputs (e.g., multiple videos/images) in a single conversation.
            """
            
            # Tokenize input conversation
            tokenized = model.build_input_ids(
                tokenizer,
                conversation,
                max_length=4096,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            # Prepare for media input (batched processing)
            all_responses = []
            
            generation_output = model.generate_caption(
                        tokenized['input_ids'].unsqueeze(0).to(model.device), 
                        tokenized['attention_mask'].unsqueeze(0).to(model.device), 
                        video_idx=tokenized['index'].unsqueeze(0),
                        video=media_tensors.unsqueeze(0), 
                        **generation_config
                    )
                
                # Decode response and append to results
            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            all_responses.append(response)

            # Combine responses
            combined_response = "\n".join(all_responses)

            # Update chat history if required
            if return_history:
                chat_history.append((user_prompt, combined_response))
                return combined_response, chat_history

            return combined_response

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
        messages = sample["messages"]
        transformed_messages = []
        conversation = ""
        

        content = messages[0]["content"]
        # Split the content into parts and identify <video> tags
        parts = content.split("<video>")
        transformed_content = []

        for i, part in enumerate(parts):
            # Add text before <video> tag
            if part.strip():
                conversation += " [INST] " + part.strip() + " [/INST]"
            # Add the video placeholder if it's not the last part
            if i < len(parts) - 1:
                conversation += "[INST] "
                conversation += ("<Video>" + VID_TOKEN + "</Video>")
                conversation += "[/INST]"
        
        

        
        path_vid1 = sample['videos'][0]

        vid1 = load_video(path_vid1, num_segments=10, return_msg=False)
        vid1 = vid1.to(model.device)

        
        
        chat_history= []
        response, chat_history = chat_multiple_media(model ,tokenizer, conversation, '', '', media_type='video', media_tensors=vid1, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
        

        append_output(sample['messages'][1]["content"], response)
        
