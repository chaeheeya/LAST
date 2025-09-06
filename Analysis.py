import pickle
import json

train_dataset = pickle.load(open('dataset/INSPIRED2/train_pred_aug_dataset_inspired2_final.pkl', 'rb'))
refined_response = json.load(open('refinement/0905224909_gpt-4.1_inspired2_train_refinement.json', 'r', encoding='utf-8'))

benchmark = []
refine = []

for sample, refined in zip(train_dataset, refined_response):
    dialog = sample['dialog']

    benchmark_resp = sample['response']
    if not benchmark_resp.strip().startswith('System: '):
        benchmark_resp = 'System: ' + benchmark_resp

    refined_resp = refined['OUTPUT']
    if not refined_resp.strip().startswith('System: '):
        refined_resp = 'System: ' + refined_resp

    benchmark.append({'DIALOG': dialog, 'RESPONSE': benchmark_resp})
    refine.append({'DIALOG': dialog, 'RESPONSE': refined_resp})
print()