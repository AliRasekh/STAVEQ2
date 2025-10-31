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
    user_message_content = "Look at the provided examples and answer the last question.\n"
    
    for i, example in enumerate(item['context_examples']):
        user_message_content += f"Example {i+1} - <video> " \
            f"The action happening in this video is: {example['template']}.\n"

    user_message_content += "<video> now considering the previous examples, " \
        "what action is happening in this video?"

    assistant_message_content = item['template']

    video_paths = [
        f"{video_directory}/{example['id']}.mp4"
        for example in item['context_examples']
    ]
    video_paths.append(f"{video_directory}/{item['id']}.mp4")

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


def generate_dataset_with_context(number_of_context=1):
    with open(annotations_path, 'r') as f:
        all_annotations = json.load(f)

    # Filter only selected classes
    filtered_annotations = [
        item for item in all_annotations if item['template'] in selected_classes
    ]

    print(f"Total filtered samples: {len(filtered_annotations)}")


    for i, annotation in enumerate(filtered_annotations):
        context_candidates = [ex for j, ex in enumerate(filtered_annotations) if j != i]
        context_examples = random.sample(context_candidates, min(number_of_context, len(context_candidates)))

        annotation['context_examples'] = [
            {
                'id': ex['id'],
                'template': ex['template'],
                'label': ex['label'],
                'placeholders': ex['placeholders']
            }
            for ex in context_examples
        ]

    test_formatted = [convert_to_sharegpt_format(item) for item in filtered_annotations]

    

    with open("test.json", 'w') as f:
        json.dump(test_formatted, f, indent=4)



generate_dataset_with_context(number_of_context=1)
