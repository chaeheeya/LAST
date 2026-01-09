import os
import time
import pickle
import json
import argparse
from datetime import datetime

from google import genai
from pytz import timezone
from tqdm import tqdm

D2R = """Pretend you are a conversational recommender system. 
I will provide you a dialog between a user and the system. 

Here is the dialog.
%s

Create a response that the system should provide.
The generated response should not exceed 100 tokens.

### Response:"""


D2R_with_item_for_train = """Pretend you are a conversational recommender system. 
I will provide you a dialog between a user and the system. 

Here is the dialog.
%s

Create a response in which the system recommends the item the user would prefer, along with relevant explanations.
(The recommended item is %s.)

When mentioning any movie or item, write its name followed by its release year in parentheses (e.g., Inception (2010)).
The generated response should not exceed 100 tokens.

### Response:"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--gemini_model', type=str, default="gemini-2.5-pro")
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--log_name', type=str, default="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    client = genai.Client(api_key=args.access_token)

    dataset_path = os.path.join('dataset', args.dataset)
    dataset = pickle.load(open(dataset_path, 'rb'))

    prompt = D2R_with_item_for_train
    # MODEL = genai.GenerativeModel(args.gemini_model)

    instructions = []
    for idx, data in enumerate(dataset):
        instruction = prompt % (data['dialog'], data['topic'])
        instructions.append(instruction)

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    if args.log_name == '':
        log_name = f'response_gen/{mdhm}_{args.gemini_model}_result.json'
    else:
        log_name = f'response_gen/{mdhm}_{args.gemini_model}_{args.log_name}'

    args.log_file = open(os.path.join(log_name), 'a', buffering=1, encoding='UTF-8')

    # Evaluation 시작
    print('GEMINI START')
    cnt = args.cnt
    # for instruction in tqdm(instructions[cnt:], bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
    pbar = tqdm(total=len(instructions), initial=cnt, bar_format=' {percentage:3.0f} % | {bar:30} {n_fmt}/{total_fmt}')
    while cnt < len(instructions):
        instruction = instructions[cnt]
        try:
            response = client.models.generate_content(model=args.gemini_model, contents=instruction)
            response = response.text

            args.log_file.write(
                json.dumps({'INPUT': instruction, 'OUTPUT': response}, ensure_ascii=False, indent=4) + '\n')
            cnt += 1
            pbar.update(1)

        except Exception as error:
            print("%s | ERROR cnt: %d" % (error, cnt))
            print(args.log_file)
            # args.cnt = int(cnt)
            time.sleep(5)
            # break

        # response = client.models.generate_content(
        #     model=args.gemini_model,
        #     contents=instruction
        # )
        #
        # args.log_file.write(json.dumps({'INPUT': instruction, 'OUTPUT': response.text}, ensure_ascii=False, indent=4) + '\n')
        # cnt += 1
