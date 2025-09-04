import os
import time
import pickle
import json
import argparse
from datetime import datetime

from openai import OpenAI
from pytz import timezone
from tqdm import tqdm


REFINE_RESPONSE_prompt = """I will give you a dialog between a user and the system, the system's next response, and an evaluation of that response in terms of informativeness, fluency, and relevance.

Dialog:
%s

Response:
%s

Response Evaluation:
%s

The evaluator's assessment criteria are:
1) Informativeness: Whether the response incorporates rich knowledge
2) Fluency: Whether the response includes diverse words
3) Relevance: Whether the response provides explanations about the recommended item with its features relevant to the dialog context

Based on the response evaluation, your task is to generate a refined response that can achieve the best possible evaluation across informativeness, fluency, and relevance.
The generated response should not exceed 100 tokens."""




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--log_name', type=str, default="")
    args = parser.parse_args()
    return args


def execute(args,
            instructions: list = None,
            labels: list = None,
            inputs: list = None):
    # openai.api_key = args.access_token
    client = OpenAI(api_key=args.access_token)

    cnt = args.cnt
    for instruction in tqdm(instructions[cnt:], bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):

        # content = template % (instruction)
        # content = mentioned_reviews_concat + refinement_template % instruction
        # print()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user",
                     # "content": review_summary_template % (label, instruction, label)}
                     "content": instruction}
                ],
                temperature=0,
            )

            response = response.choices[0].message.content

            args.log_file.write(json.dumps({'INPUT': instruction, 'OUTPUT': response}, ensure_ascii=False, indent=4) + '\n')
            cnt += 1

        except Exception as error:
            print("ERROR cnt: %d" % (cnt))
            print(args.log_file)
            args.cnt = int(cnt)
            time.sleep(5)
            break

        # openai.api_requestor._thread_context.session.close()
        if int(cnt) == len(instructions):
            return False



def chatgpt_test(args,
                 instructions: list = None,
                 labels: list = None,
                 inputs: list = None,
                 ):
    print('CHATGPT_TEST_START')
    while True:
        if execute(args=args, instructions=instructions, labels=labels, inputs=inputs) == False:
            break



if __name__ == "__main__":

    args = parse_args()

    inspired2_train = pickle.load(open('dataset/INSPIRED2/train_pred_aug_dataset_inspired2_final.pkl', 'rb'))
    inspired2_test = pickle.load(open('dataset/INSPIRED2/test_pred_aug_dataset_inspired2_final.pkl', 'rb'))

    generated_response = json.load(open('response_gen/0904190453_gpt-4.1_inspired2_test_GPT_response_100token.json', 'r', encoding='utf-8'))
    eval_gen_response = json.load(open('evaluation/gpt_eval/0904192110_gpt-4.1_result_inspired2_test_gpt-4.1-response_100token.json', 'r', encoding='utf-8'))

    prompt = REFINE_RESPONSE_prompt
    MODEL = "gpt-4.1"

    instructions = []
    for idx, (data, gen_response, eval_gen) in enumerate(zip(inspired2_test, generated_response, eval_gen_response)):
        dialog = data['dialog']
        new_response = gen_response['OUTPUT']
        if not new_response.strip().startswith('System: '):
            new_response = 'System: ' + new_response

        eval_response = eval_gen['OUTPUT'].split('<think>')[-1].split('</think>')[0].strip()

        instruction = prompt % (dialog, new_response, eval_response)
        instructions.append(instruction)

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    if args.log_name == '':
        log_name = f'refinement/{mdhm}_{MODEL}_inspired2_refinement.json'
    else:
        log_name = args.log_name

    args.log_file = open(os.path.join(log_name), 'a', buffering=1, encoding='UTF-8')

    chatgpt_test(args, instructions=instructions)
