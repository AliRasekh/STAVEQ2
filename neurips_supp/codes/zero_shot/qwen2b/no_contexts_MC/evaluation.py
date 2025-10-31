from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)



def evaluate_answer(ground_truth, llm_answer):
    prompt = (
        f"""
        Ground Truth: {ground_truth}
        LLM Answer: {llm_answer}

        Based on the ground truth, is the LLM answer correct? Answer with a simple "Yes" or "No".

        """
    )
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    evaluation = response.strip().lower()
    return "yes" in evaluation

def extract_labels_and_responses(file_path):
    ground_truths = []
    responses = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        current_label = None
        current_response = []
        capturing_response = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Label:"):
                current_label = line.replace("Label: ", "")
            
            elif line.startswith("Response:"):
                current_response = [line.replace("Response: ", "")]
                capturing_response = True
            
            elif line == "-" * 40:
                if current_label and current_response:
                    ground_truths.append(current_label)
                    responses.append("\n".join(current_response))
                
                current_label = None
                current_response = []
                capturing_response = False
            
            elif capturing_response:
                current_response.append(line)
    
    return ground_truths, responses


file_path = 'output.txt'
ground_truths, llm_answers = extract_labels_and_responses(file_path)

evaluation_results = []
for ground_truth, llm_answer in zip(ground_truths, llm_answers):
    result = evaluate_answer(ground_truth, llm_answer)
    evaluation_results.append(result)

with open('result.txt', "w") as file:
    for i, result in enumerate(evaluation_results):
        file.write('True\n' if result else 'False\n')

    correct_answers = sum(evaluation_results)
    total_answers = len(evaluation_results)
    accuracy_percentage = (correct_answers / total_answers) * 100
    print(f"Accuracy: {accuracy_percentage:.2f}%")
    file.write(f"Accuracy: {accuracy_percentage:.2f}%")