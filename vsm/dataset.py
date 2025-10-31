import json, os
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.io as io
from transformers import AutoProcessor
from torchvision import transforms
from decord import VideoReader, cpu
from decord import bridge
bridge.set_bridge('torch')

train_annotations_path = 'subsets/train.json'
test_annotations_path = 'subsets/test.json'
data_path = '../data/ssv2/data_mp4'


class SSv2(Dataset):

    def __init__(self, test=False):

        super().__init__()

        self.test = test
        annotations_path = test_annotations_path if test else train_annotations_path
        with open(annotations_path, 'r') as f:
            self.annotations = json.loads(f.read())

        min_pixels = (2 * 28) * (2 * 28)
        max_pixels = (5 * 28) * (8 * 28)
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

        video_inputs = self.read_video(video_paths)
        return video_inputs


    def read_video(self, paths):
        num_frames_to_sample = 8
        all_videos = []
        shapes = []

        for path in paths:
            video_reader = io.VideoReader(path, "video")
            metadata = video_reader.get_metadata()["video"]
            source_fps = metadata['fps'][0]
            duration = metadata['duration'][0]

            sample_times = np.linspace(0, duration - 1/source_fps, num_frames_to_sample)
            indices = np.round(sample_times * source_fps).astype(np.int64)

            frames = []
            for i, frame in enumerate(video_reader):
                if i in indices:
                    frame_tensor = frame['data']
                    frames.append(frame_tensor)
                if len(frames) == num_frames_to_sample:
                    break

            video_tensor = torch.stack(frames)
            all_videos.append(video_tensor)
            shapes.append((video_tensor.shape[2], video_tensor.shape[3]))

        min_h = min(h for h, w in shapes)
        min_w = min(w for h, w in shapes)

    
        cropped_videos = []
        for video in all_videos:
            _, _, h, w = video.shape
            top = (h - min_h) // 2
            left = (w - min_w) // 2
            cropped = video[:, :, top:top + min_h, left:left + min_w]

            cropped_videos.append(cropped)

        return cropped_videos

