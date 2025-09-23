import os
import time
import pickle
import json
import argparse

import anthropic

from datetime import datetime
from pytz import timezone
from tqdm import tqdm

D2R = """Pretend you are a conversational recommender system. 
I will provide you a dialog between a user and the system. 

Here is the dialog.
%s

Create a response that the system should provide.
The generated response should not exceed 100 tokens.

### Response:"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--claude_model', type=str, default="claude-sonnet-4-20250514")
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--log_name', type=str, default="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    client = anthropic.Anthropic(api_key=args.access_token)

    dataset = pickle.load(open(args.dataset_path, 'rb'))
    prompt = D2R
    # MODEL = genai.GenerativeModel(args.gemini_model)

    instructions = []
    for idx, data in enumerate(dataset):
        instruction = prompt % (data['dialog'])
        instructions.append(instruction)

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    if args.log_name == '':
        log_name = f'response_gen/{mdhm}_{args.claude_model}_result.json'
    else:
        log_name = args.log_name

    args.log_file = open(os.path.join(log_name), 'a', buffering=1, encoding='UTF-8')

    # Evaluation 시작
    print('CLAUDE START')
    cnt = args.cnt
    # for instruction in tqdm(instructions[cnt:], bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
    pbar = tqdm(total=len(instructions), initial=cnt, bar_format=' {percentage:3.0f} % | {bar:30} {n_fmt}/{total_fmt}')
    while cnt < len(instructions):
        instruction = instructions[cnt]
        try:
            response = client.messages.create(
                model=args.claude_model,
                messages=[
                    {"role": "user", "content": instruction}
                ],
                temperature=0,
                max_tokens=100
            )
            response = response.content[0].text

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
