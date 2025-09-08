import json

eval_result_path = 'evaluation/gpt_eval/gpt-4.1_eval_refined_gpt-0906222819_inspired2_test_LLaMA-3.1-response_train_only_resp_sft.json'
eval_result = json.load(open(eval_result_path, 'r', encoding='utf-8'))

informativeness, fluency, relevance = 0, 0, 0
for result in eval_result:

    output = result['OUTPUT']
    think = output.split('<think>')[-1].split('</think>')[0].strip()
    answer = output.split('<answer>')[-1].split('</answer>')[0].strip()

    dict_answer = json.loads(answer)
    for key, value in dict_answer.items():
        if 'inform' in key.lower():
            informativeness += int(value)
        if 'fluen' in key.lower():
            fluency += int(value)
        if 'relevan' in key.lower():
            relevance += int(value)


print(f"informativeness: {round(informativeness/len(eval_result), 2)}")
print(f"fluency: {round(fluency/len(eval_result), 2)}")
print(f"relevance: {round(relevance/len(eval_result), 2)}")

print()