import random
import json
import os

selected_classes = [
    "Pulling [something] from left to right",
    "Pulling [something] from right to left",
    "Throwing [something] in the air and catching it",
    "Throwing [something] in the air and letting it fall",
    "[Something] falling like a rock",
    "Rolling [something] on a flat surface",
    "Poking a stack of [something] so the stack collapses",
    "Picking [something] up",
    "Moving [something] away from [something]",
    "Moving [something] closer to [something]"
]

annotations_path = '../data/ssv2/labels/train.json'
video_directory = '../data/ssv2/data_mp4'


def convert_to_format(item):

    assistant_message_content = \
        f"The action happening in this video is: {item['template']}"

    output = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Look at the provided video and answer the question."
                },
                {
                    "type": "video",
                    "video": f"{item['id']}.mp4"
                },
                {
                    "type": "text",
                    "text": "What is the action happening in this video?"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_message_content
                }
            ]
        }
    ]

    return output


def create_T10(test_ratio=0.2):

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    filtered_annotations = [
        item for item in annotations if item['template'] in selected_classes
    ]

    random.shuffle(filtered_annotations)

    split_index = int(len(filtered_annotations) * (1 - test_ratio))
    train_annotations = filtered_annotations[:split_index]
    test_annotations = filtered_annotations[split_index:]
    train_sharegpt = [convert_to_format(item) for item in train_annotations]
    test_sharegpt = [convert_to_format(item) for item in test_annotations]

    os.makedirs("subsets", exist_ok=True)

    with open("subsets/train.json", 'w') as f:
        json.dump(train_sharegpt, f, indent=4)

    with open("subsets/test.json", 'w') as f:
        json.dump(test_sharegpt, f, indent=4)


create_T10()
