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

annotations_path = '../../../data/ssv2/labels/train.json'
video_directory = '../../../data/ssv2/data_mp4'

def convert_to_sharegpt_format(item):
    user_message_content = "<video> From the following actions, which one is happening in the video?\nPulling [something] from left to right\nPulling [something] from right to left\nThrowing [something] in the air and catching it\nThrowing [something] in the air and letting it fall\n[Something] falling like a rock\nRolling [something] on a flat surface\nPoking a stack of [something] so the stack collapses\nPicking [something] up\nMoving [something] away from [something]\nMoving [something] closer to [something]\n"
    assistant_message_content = item['template']

    video_paths = [f"{video_directory}/{item['id']}.mp4"]

    output = {
        "messages": [
            {
                "content": user_message_content,
                "role": "user"
            },
            {
                "content": assistant_message_content,
                "role": "assistant"
            }
        ],
        "videos": video_paths
    }

    return output

def create_T10():
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    
    filtered_annotations = [
        item for item in annotations if item['template'] in selected_classes
    ]

    random.shuffle(filtered_annotations)


    test_sharegpt = [convert_to_sharegpt_format(item) for item in filtered_annotations]


    with open("subsets/test.json", 'w') as f:
        json.dump(test_sharegpt, f, indent=4)


create_T10()
