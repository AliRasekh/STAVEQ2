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

def to_qwen_format(item):
    content = []

    # Add context videos
    for i, ctx in enumerate(item['context_examples']):
        ctx_path = f"{video_directory}/{ctx['id']}.mp4"
        video_content = {
            "type": "video",
            "video": ctx_path,
            "max_pixels": 240 * 420,
            "fps": 1.0,
        }
        text_content = {
            "type": "text",
            "text": f"This is video number {i}, and the action happening in it is {ctx['template']}.\n"
        }
        content.append(video_content)
        content.append(text_content)

    # Add target video
    video_path = f"{video_directory}/{item['id']}.mp4"
    video_content = {
        "type": "video",
        "video": video_path,
        "max_pixels": 360 * 420,
        "fps": 1.0,
    }

    if len(item['context_examples']) > 0:
        text_content = {
            "type": "text",
            "text": "This is the last video. What action is happening in the last video?"
        }
    else:
        text_content = {
            "type": "text",
            "text": "From the following actions, which one is happening in the video?\nPulling [something] from left to right\nPulling [something] from right to left\nThrowing [something] in the air and catching it\nThrowing [something] in the air and letting it fall\n[Something] falling like a rock\nRolling [something] on a flat surface\nPoking a stack of [something] so the stack collapses\nPicking [something] up\nMoving [something] away from [something]\nMoving [something] closer to [something]\n"
        }

    content.append(video_content)
    content.append(text_content)

    return [{"role": "user", "content": content}]

def generate_random_context_dataset(num_context=5, test_ratio=0.2):
    with open(annotations_path, 'r') as f:
        all_annotations = json.load(f)


    filtered = [a for a in all_annotations if a['template'] in selected_classes]
    print(f"Total samples from selected classes: {len(filtered)}")


    for i, item in enumerate(filtered):
        other_candidates = [x for j, x in enumerate(filtered) if j != i]
        context_samples = random.sample(other_candidates, min(num_context, len(other_candidates)))

        item['context_examples'] = [
            {
                'id': ctx['id'],
                'template': ctx['template'],
                'label': ctx['label'],
                'placeholders': ctx['placeholders']
            }
            for ctx in context_samples
        ]

    random.shuffle(filtered)
    


    test_formatted = [to_qwen_format(x) for x in filtered]


    with open("subsets/test.json", "w") as f:
        json.dump(test_formatted, f, indent=4)



generate_random_context_dataset(num_context=0)