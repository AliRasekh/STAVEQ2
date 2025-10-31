import random
import json, os

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


def convert_to_sharegpt_format_positive_negative(item):
    messages = []

    user_content = []

    user_content.append({
        "type": "text",
        "text": "Look at the provided examples and identify which example is related to the final video."
    })

    for i, example in enumerate(item['context_examples']):
        user_content.append({
            "type": "text",
            "text": f"Example {i+1} -"
        })
        user_content.append({
            "type": "video",
            "video": f"{example['id']}.mp4"
        })


    user_content.append({
        "type": "text",
        "text": "Now consider the previous examples. Is there any action related to this video?"
    })
    user_content.append({
        "type": "video",
        "video": f"{item['id']}.mp4"
    })
    user_content.append({
        "type": "text",
        "text": "If not, respond with 'No related action'. If there is, respond with the example number."
    })


    if item.get('is_positive', False):
        related_index = item['context_examples'].index(
            next(e for e in item['context_examples'] if e['template'] == item['template'])
        )
        assistant_content = [{
            "type": "text",
            "text": f"The related example is Example {related_index + 1}"
        }]
    else:
        assistant_content = [{
            "type": "text",
            "text": "No related action"
        }]

    output = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]

    return output



def select_subset_with_positive_negative(selection_percent=80, positive_percent=80, number_of_context=2):

    result_annotations_path = f"subsets/test.json"

    with open(annotations_path, 'r') as f:
        annotations = json.loads(f.read())

    with open('subsets/train.json', 'r') as f:
        filtered = json.loads(f.read())

    excluded_video_ids = set(
    os.path.splitext(os.path.basename(content['video']))[0]
    for item in filtered
    for message in item
    if message['role'] == 'user'
    for content in message['content']
    if content['type'] == 'video'
    )



    annotations_by_class = {
        # selected_class: [item for item in annotations if item['template'] == selected_class]
        selected_class: [item for item in annotations if item['template'] == selected_class and item['id'] not in excluded_video_ids]
        for selected_class in selected_classes
    }

    samples = []
    for selected_class, class_annotations in annotations_by_class.items():
        # total_sample_size = max(1, int(len(class_annotations) * (selection_percent / 100)))
        selected_samples = class_annotations
        total_sample_size = len(class_annotations)

        positive_sample_size = max(1, int(total_sample_size * (positive_percent / 100)))
        positive_samples = selected_samples[:positive_sample_size]
        negative_samples = selected_samples[positive_sample_size:]

        for sample in positive_samples:
            related_context = random.choice([
                item for item in class_annotations if item['id'] != sample['id']
            ])
            unrelated_class = random.choice([
                cls for cls in selected_classes if cls != selected_class
            ])
            unrelated_context = random.choice(annotations_by_class[unrelated_class])

            context_examples = [related_context, unrelated_context]
            random.shuffle(context_examples)

            sample['context_examples'] = [
                {
                    'id': example['id'],
                    'template': example['template'],
                    'label': example['label'],
                    'placeholders': example['placeholders'],
                }
                for example in context_examples
            ]
            sample['is_positive'] = True
            samples.append(sample)

        for sample in negative_samples:
            unrelated_contexts = random.sample(
                [item for item in annotations if item['template'] != sample['template']],
                number_of_context
            )
            sample['context_examples'] = [
                {
                    'id': example['id'],
                    'template': example['template'],
                    'label': example['label'],
                    'placeholders': example['placeholders'],
                }
                for example in unrelated_contexts
            ]
            sample['is_positive'] = False
            samples.append(sample)

    random.shuffle(samples)
    sharegpt_annotations = [
        convert_to_sharegpt_format_positive_negative(item) for item in samples
    ]

    with open(result_annotations_path, 'w') as f:
        f.write(json.dumps(sharegpt_annotations, indent=4))

select_subset_with_positive_negative(positive_percent=80)
