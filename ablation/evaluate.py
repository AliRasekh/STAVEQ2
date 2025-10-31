def calculate_accuracy(file_path):

    correct = 0
    total = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break

        llm_line = lines[i+1].strip()
        gt_line = lines[i+2].strip()

        llm_pred = llm_line.replace("LLM: ", "").replace("<|im_end|>", "").strip()
        gt_text = gt_line.replace("GT: ", "").strip()

        if llm_pred == gt_text:
            correct += 1

        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"Total comparisons: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
