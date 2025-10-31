import random, csv
import json, os

# inputs
annotations_path1 = '../data/ssv2/labels/train.json'


# outputs
subset_directory = 'subsets'


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


def select_subset_train():

    train_annotations_path = os.path.join(
        subset_directory, f"train.json"
    )

    if not os.path.exists(subset_directory):
        os.makedirs(subset_directory)

    with open(annotations_path1, 'r') as f:
        annotations = json.loads(f.read())

    random.shuffle(annotations)

    train_annotations = [
        convert_to_format(annotations[i])
        for i in range(len(annotations))
    ]

    with open(train_annotations_path, 'w') as f:
        f.write(json.dumps(train_annotations, indent=4))



select_subset_train()



annotations_path = '../data/ssv2/labels/test.json'
template_path = '../data/ssv2/labels/test-answers.csv'




def load_templates(template_path):
    templates = {}
    with open(template_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_split = row[0].split(';')
            video_id, template = row_split[0], row_split[1]
            templates[video_id] = template
    return templates


def select_subset_test():
    templates = load_templates(template_path)

    test_annotations_path = os.path.join(subset_directory, f"test.json")

    if not os.path.exists(subset_directory):
        os.makedirs(subset_directory)

    with open(annotations_path, 'r') as f:
        annotations = json.loads(f.read())

    class_annotations = {}

    for item in annotations:
        item_id = item['id']
        if item_id in templates:
            template = templates[item_id]
            class_annotations_temp = {"id": item_id, "template": template}
            class_annotations.append(class_annotations_temp)



    class_annotations = [convert_to_format(item) for item in class_annotations]

    with open(test_annotations_path, 'w') as f:
        json.dump(class_annotations, f, indent=4)


select_subset_test()
