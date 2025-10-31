import json, os
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.io as io
from transformers import AutoProcessor

train_annotations_path = 'subsets/train.json'
test_annotations_path = 'subsets/test.json'
data_path = '../data/ssv2/data_mp4'

min_pixels = 16 * (28 * 28)
max_pixels = 64 * (28 * 28)


class SSv2(Dataset):

    def __init__(self, test=False):

        super().__init__()

        self.test = test
        annotations_path = test_annotations_path if test else train_annotations_path
        with open(annotations_path, 'r') as f:
            self.annotations = json.loads(f.read())

        self.preprocessor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.response_pattern = self.preprocessor.tokenizer.encode('<|im_start|>assistant\n')
        self.visual_tokens = [151652, 151653, 151656]


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        return self.annotations[idx]

    
    def collate(self, examples):

        video_inputs = []
        text_inputs = []

        for example in examples:

            video_inputs.extend(self.get_videos(example))
            text_inputs.append(
                self.preprocessor.apply_chat_template(
                    example[:-1] if self.test else example,
                    add_generation_prompt=self.test,
                    tokenize=False
                )
            )

        inputs = self.preprocessor(
            text=text_inputs,
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if not self.test:
            inputs["labels"] = self.get_labels(inputs)

        return inputs


    def get_labels(self, inputs):

        labels = inputs["input_ids"].clone()

        labels[labels == self.preprocessor.tokenizer.pad_token_id] = -100
        for visual_token_id in self.visual_tokens:
            labels[labels == visual_token_id] = -100
        
        for pos in torch.nonzero(labels == self.response_pattern[0]):
            if (
                self.response_pattern ==
                labels[pos[0], pos[1]:pos[1]+len(self.response_pattern)].tolist()
            ):
                labels[pos[0], :pos[1]+len(self.response_pattern)] = -100

        return labels


    def get_videos(self, item):

        video_paths = []

        for conv in item:
            for content in conv["content"]:
                if content["type"] == "video":
                    path = os.path.join(data_path, content["video"])
                    video_paths.append(path)

        # image_inputs, video_inputs = process_vision_info(item)
        video_inputs = [self.read_video(path) for path in video_paths]
        return video_inputs


    def read_video(self, path, fps=2, min_frames=2, max_frames=8):

        video_reader = io.VideoReader(path, "video")
        source_fps = video_reader.get_metadata()["video"]['fps'][0]
        duration = video_reader.get_metadata()["video"]['duration'][0]
        # total_frame_count = int(source_fps * duration)

        sample_times = np.arange(0, duration - 1/source_fps, 1/fps)
        if len(sample_times) < min_frames:
            sample_times = np.linspace(0, duration - 1/source_fps, min_frames)

        sample_size = min(len(sample_times) // 2 * 2, max_frames)
        sample_times = sample_times[:sample_size]
        indices = np.round(sample_times * source_fps).astype(np.int64)

        frames = []
        for i, frame in enumerate(video_reader):
            if i in indices:
                frames.append(frame['data'])

        frames = torch.stack(frames)
        return frames
