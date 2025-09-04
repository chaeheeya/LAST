import pickle
import json


def calculate_score(evaluation):
    informativeness, relevance, fluency = 0, 0, 0

    answer = evaluation.split('<answer>')[-1].split('</answer>')[0].strip()

    dict_answer = json.loads(answer)
    for key, value in dict_answer.items():
        if 'inform' in key.lower():
            informativeness += int(value)
        if 'fluen' in key.lower():
            fluency += int(value)
        if 'relevan' in key.lower():
            relevance += int(value)

    return informativeness+relevance+fluency

if __name__=='__main__':

    original = json.load(open('response_gen/0903192711_gpt-4.1_inspired2_train_GPT_response.json', 'r', encoding='UTF-8'))
    original_eval = json.load(open('evaluation/gpt_eval/0903233707_gpt-4.1_result_inspired2_train_gpt4.1-response_scoringP.json', 'r', encoding='UTF-8'))

    refined = json.load(open('refinement/0904104837_gpt-4.1_inspired2_train_refinement.json', 'r', encoding='UTF-8'))
    refined_eval = json.load(open('evaluation/gpt_eval/0904131604_gpt-4.1_result_inspired2_train_refined-response_scoringP.json', 'r', encoding='UTF-8'))

    for x, x_eval, y, y_eval  in zip(original, original_eval, refined, refined_eval):
        dialog = refined['INPUT'].split('Dialog:')[-1].split('Response:')[0].strip()
        prev_response = refined['INPUT'].split('Response:')[-1].split('Response Evaluation:')[0].strip()
        refined_response = refined['OUTPUT']
        if not refined_response.strip().startswith('System: '):
            refined_response = 'System: ' + refined_response

        original_score = calculate_score(x_eval['OUTPUT'])
        refined_score = calculate_score(y_eval['OUTPUT'])

        if original_score <= refined_score:
