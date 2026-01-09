
import os
import time
import random
import pickle
import json
import argparse
from datetime import datetime

from openai import OpenAI
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


D2R_with_item_for_test = """Pretend you are a conversational recommender system. 
I will provide you a dialog between a user and the system. 

Here is the dialog.
%s

Create a response in which the system recommends the item the user would prefer, along with relevant explanations.

The recommended item and response are enclosed within <item></item> and <answer></answer> tags, respectively, i.e., <item>recommended item here</item>\n<answer>response here</answer>

When mentioning any movie or item, write its name followed by its release year in parentheses (e.g., Inception (2010)).
The generated response should not exceed 100 tokens..

### Response:"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--log_name', type=str, default="")
    parser.add_argument('--dataset', type=str, default="")

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
            kwargs = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": instruction
                    }
                ],
            }

            # GPT-4 계열만 temperature=0 사용
            if not MODEL.startswith("gpt-5"):
                kwargs["temperature"] = 0

            response = client.chat.completions.create(**kwargs)

            # response = client.chat.completions.create(
            #     model=MODEL,
            #     messages=[
            #         {"role": "user",
            #          # "content": review_summary_template % (label, instruction, label)}
            #          "content": instruction}
            #     ],
            #     temperature=0,
            # )

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

    MODEL = args.model # gpt-4.1-mini / gpt-4.1 / gpt-5-mini / gpt-5.2
    prompt = D2R_with_item_for_train

    instructions = []
    dataset_path = os.path.join('dataset', args.dataset)
    dataset = pickle.load(open(dataset_path, 'rb'))

    for idx, data in enumerate(dataset):
        dialogs = data['dialog'].split('\n')[-6:]
        dialog = ''
        for d in dialogs:
            dialog += d
            dialog += '\n'
        dialog = dialog.strip()

        instruction = prompt % (dialog, data['topic'])
        instructions.append(instruction)


    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    if args.log_name == '':
        log_name = f'response_gen/{mdhm}_{MODEL}_inspired2_test_GPT_response_100token.json'
    else:
        log_name = f'response_gen/{mdhm}_{args.log_name}.json'

    args.log_file = open(os.path.join(log_name), 'a', buffering=1, encoding='UTF-8')

    chatgpt_test(args, instructions=instructions)
