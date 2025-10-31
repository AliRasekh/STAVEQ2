import random
import json


def convert_to_qwen_format(item, path_prefix):

    if len(item["history"]) != 0:
        print("ignoring history")

    output = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"{path_prefix}/{item['videoid']}.mp4"
                },
                {
                    "type": "text",
                    "text": item["question"]
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": item["answer"]
                }
            ]
        }
    ]

    return output


def switch_options(item):

    question = item[0]['content'][1]['text']
    option_a_start = question.find("(A)")
    option_b_start = question.find("(B)")

    question_only = question[:option_a_start-1]
    option_a = question[option_a_start+4:option_b_start-1]
    option_b = question[option_b_start+4:]
    
    answer = item[1]['content'][0]['text']
    correct_option = answer[:3]
    answer_only = answer[4:]

    new_question = f"{question_only} (A) {option_b} (B) {option_a}"

    if correct_option == "(A)":
        new_answer = f"(B) {answer_only}"
    else:
        new_answer = f"(A) {answer_only}"
    
    item[0]['content'][1]['text'] = new_question
    item[1]['content'][0]['text'] = new_answer


with open('subsets/webvid_qa.json', 'r') as f:
    annotations = json.loads(f.read())

dataset = [
    convert_to_qwen_format(item, '../data/webvid/data_mp4') for item in annotations
]



for item in dataset:
    if random.choice([True, False]):
        switch_options(item)

random.shuffle(dataset)

with open('subsets/final_dataset.json', 'w') as f:
    f.write(json.dumps(dataset, indent=4))

