import json

eval_result_path = 'evaluation/gemini_eval/gemini-2.5-pro_eval_inspired2_test_claude-sonnet-resp_twodimP.json'
eval_result = json.load(open(eval_result_path, 'r', encoding='utf-8'))

validity, informativeness, fluency, relevance = 0, 0, 0, 0
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
        if 'valid' in key.lower():
            validity += int(value)


print(f"validity: {round(validity/len(eval_result), 2)}")
print(f"informativeness: {round(informativeness/len(eval_result), 2)}")
print(f"fluency: {round(fluency/len(eval_result), 2)}")
print(f"relevance: {round(relevance/len(eval_result), 2)}")

print()