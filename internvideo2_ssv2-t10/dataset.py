import json, os
import numpy as np
from transformers import AutoTokenizer, AutoProcessor
import torch
from torch.utils.data.dataset import Dataset
# from qwen_vl_utils import process_vision_info
import decord
from transformers import AutoTokenizer
from decord import VideoReader, cpu
from torchvision import transforms
decord.bridge.set_bridge("torch")

tokenizer =  AutoTokenizer.from_pretrained(
    'OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)

processor = AutoProcessor.from_pretrained(
    'OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)


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

annotations_path = 'subsets/train.json'
data_path = '../data/ssv2/data_mp4'


def build_input_ids(
    tokenizer, 
    conversation,
    max_length,
    add_special_tokens,
    truncation,
    image = None, 
    video = None, 
    padding = "longest", 
    return_tensors = "pt",
    image_placeholder: str = DEFAULT_IMG_PLACEHOLDER,
    video_placeholder: str = DEFAULT_VID_PLACEHOLDER,
):

    input_ids = []
    indexs = []
    attention_mask = []
    start, total_len = 0, 0
    while True:
        index1 = conversation.find(image_placeholder, start)
        index2 = conversation.find(video_placeholder, start)
        if index1 == -1 and index2 == -1:
            index = -1
        elif index1 == -1:
            index = index2
        elif index2 == -1:
            index = index1
        else:
            index = min(index1, index2)
            assert index != -1
        if index == -1:
            inputs = tokenizer(conversation[start:], max_length=max_length-total_len, truncation=truncation, padding=padding, return_tensors=return_tensors)
        else:
            inputs = tokenizer(conversation[start:index], max_length=max_length,  truncation=truncation, padding='longest', return_tensors=return_tensors)
        
        input_ids += inputs.input_ids
        attention_mask += inputs.attention_mask
        total_len += inputs.input_ids[0].shape[0]
        indexs += torch.zeros_like(inputs.input_ids)
        
        if index != -1:
            input_ids += [torch.zeros(96).long()]
            attention_mask += [torch.ones(96).long()]
            indexs += [torch.ones(96)]
        
        if index == -1:
            return {
                'input_ids': torch.cat(input_ids),
                'attention_mask': torch.cat(attention_mask),
                'index': torch.cat(indexs).to(torch.bool),
            }
        start = index + len(DEFAULT_IMG_PLACEHOLDER)



def collect_input(data, media_tensor):
    result = ""
    
    for entry in data:
        if entry["role"] == "user":
            for content in entry["content"]:
                if content["type"] == "text":
                    result += f"[INST] {content['text']} [/INST] "
                elif content["type"] == "video":
                    result += "[INST] <Video>[<VID_PLH>]</Video> [/INST] "
        elif entry["role"] == "assistant":
            for content in entry["content"]:
                if content["type"] == "text":
                    result += f"{content['text']}</s>"
                    
    # print(result)
    
    # tokenized = build_input_ids(
    #         tokenizer,
    #         result,
    #         max_length=248,
    #         add_special_tokens=True,
    #         truncation=False,
    #         padding=False,
    #         return_tensors='pt'
    #     )
    # return

    tokenized = build_input_ids(
            tokenizer,
            result,
            max_length=248,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        

    inputs = {
        'input_ids': tokenized['input_ids'].unsqueeze(0),
        'attention_mask': tokenized['attention_mask'].unsqueeze(0),
        "video_idx": tokenized['index'].unsqueeze(0),
        "video": media_tensor.unsqueeze(0),
        # **generation_config
    }


    return inputs
    

def mask_tensor(tensor):
    last_four_index = (tensor == 4).nonzero(as_tuple=True)[-1][-1].item() + 1
    
    masked_tensor = torch.where(torch.arange(tensor.numel()) < last_four_index, -100, tensor.view(-1))
    
    return masked_tensor.view(tensor.shape)


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_path, num_segments=8, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
    frames = transform(frames)
    return frames


def process_video(example):
        video_paths = []

        for conv in example:
            for content in conv["content"]:
                if content["type"] == "video":
                    path = os.path.join(data_path, content["video"])
                    video_paths.append(path)
        
        
        video_tensor = load_video(video_paths[0])

        
        return video_tensor


class SSv2(Dataset):

    def __init__(self):

        self.response_pattern = tokenizer.encode('video? [/INST]')

        with open(annotations_path, 'r') as f:
            self.annotations = json.loads(f.read())


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        return self.annotations[idx]

    
    def collate(self, examples):

        video_inputs = []
        text_inputs = []

        for example in examples:
            video_inputs = process_video(example)
            inputs = collect_input(example, video_inputs)
            
        labels = inputs["input_ids"].clone()

        labels = mask_tensor(labels)

        inputs["labels"] = labels

        return inputs
