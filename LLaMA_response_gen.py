import os
import pickle
import json
import argparse

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from pytz import timezone
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

instruction = """Pretend you are a conversational recommender system. 
Create a response that the system should provide."""


class Dataset_processing(Dataset):
    def __init__(self, args, train_dataset, test_dataset, tokenizer, instruction):
        self.args = args
        self.tokenizer = tokenizer

        self.instruction = instruction

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # dialog -> utterance 로 쪼개기
        for dataset in [self.train_dataset, self.test_dataset]:
            for data in dataset:
                dialog = data['dialog'].split('\n')
                context = []
                for utt in dialog:
                    if "System: " in utt:
                        utt = utt.split('System: ')[1].split('\n')[0]
                        context.append({'role': "assistant", 'content': utt})
                    elif "User: " in utt:
                        utt = utt.split('User: ')[1].split('\n')[0]
                        context.append({'role': "user", 'content': utt})
                    else:
                        print('ERROR')
                data['dialog'] = context

        # dataset format 맞추기
        if self.args.dataset == 'train':
            dataset = self.train_dataset
        elif self.args.dataset == 'test':
            dataset = self.test_dataset

        print("Dataset length: ", len(dataset))
        self.formatted_dataset = []
        for data in dataset:
            formatted_context = self.tokenizer.apply_chat_template(
                [{'role': 'system', 'content': self.instruction}] + data['dialog'][-5:],
                tokenize=False,
                add_generation_prompt=True)
            # formatted_context = self.tokenizer(formatted_context, padding='max_length', truncation=True, max_length=1024, return_tensors='pt')
            dialog_text = "\n".join([f"{i['role']}: {i['content']}" for i in data['dialog']])
            self.formatted_dataset.append(
                {'formatted_context': formatted_context, 'dialog': dialog_text, 'response': data['response']})

        # self.tokenizer.apply_chat_template([{'role': 'system', 'content': instruction}] + inspired2_train[0]['dialog'], tokenize=True, padding=True, max_length=128, add_generation_prompt=True)

    def __len__(self):
        # 데이터 샘플 개수를 반환
        return len(self.formatted_dataset)

    def __getitem__(self, idx):
        data = self.formatted_dataset[idx]
        return {'formatted_context': data['formatted_context'], "dialog": data['dialog']}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="train")
    parser.add_argument('--lora_weights', type=str, default=None)
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--log_name', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--max_new_tokens', type=int, default=100)

    args = parser.parse_args()

    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(__file__)
    elif sysChecker() == "Windows":
        args.home = ''
    print(args.home)

    if args.model_path != '':
        args.model_path = os.path.join(args.home, 'model_weights', args.model_path)

    return args


def load_base_model(model_name, model_path=''):
    device_map = {"": 0}

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("world_size: %d" % world_size)
    if world_size != 1:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        print(device_map)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name if model_path == '' else model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    return base_model


def load_peft_model(model, model_path, is_trainable=True):
    if model_path != '':
        peft_model = PeftModel.from_pretrained(model, model_path, is_trainable=is_trainable)
    else:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


if __name__ == '__main__':
    # device 설정
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        print('Set a pad token as <|pad|> in the tokenizer')
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"

    model = load_base_model("meta-llama/Llama-3.1-8B-Instruct")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.model_path:
        model = load_peft_model(model, args.model_path)

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))

    result_path = os.path.join(args.home, 'response_gen')
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    json_path = os.path.join(result_path, f'{mdhm}_inspired2_LLaMA-3.1-response.json')
    json_file = open(json_path, 'a', buffering=1, encoding='UTF-8')

    rank, world_size = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # 데이터셋 로드
    inspired2_train = pickle.load(open('dataset/INSPIRED2/train_pred_aug_dataset_inspired2_final.pkl', 'rb'))
    inspired2_test = pickle.load(open('dataset/INSPIRED2/test_pred_aug_dataset_inspired2_final.pkl', 'rb'))

    dataset = Dataset_processing(args, inspired2_train, inspired2_test, tokenizer, instruction)
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    step = 1
    for data in tqdm(dataset_loader, bar_format='{percentage:3.0f} % | {bar:23} {r_bar}'):
        batch_size = len(data['dialog'])
        inputs = tokenizer(data['formatted_context'], return_tensors='pt', padding=True)

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        generation_config = GenerationConfig(
            num_beams=1
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
        sequences = generation_output.sequences  # shape: (batch * num_return_sequences, seq_len)
        outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        outputs = [output[output.rfind('assistant\n'):].split('assistant\n')[-1].replace('\n', '').strip() for output in
                   outputs]

        for i in range(batch_size):
            dialog = data['dialog'][i]
            output = outputs[i]

            json_file.write(json.dumps({"INPUT": dialog, "OUTPUT": output}, ensure_ascii=False, indent=4) + '\n')

#     for idx in tqdm(range(0, len(inspired2_train), args.batch_size)):
#         batch = inspired2_train[idx:idx + args.batch_size]
#         batch_input_prompt = get_prompt(D2R, batch)
#
#         model.eval()
#
#         inputs = tokenizer.apply_chat_template(
#             batch_input_prompt,
#             tokenize=True,
#             add_generation_prompt=True,
#             return_tensors="pt",
#             padding=True
#         )
#
#         input_ids = inputs['input_ids'].to(device)
#         attention_mask = inputs['attention_mask'].to(device)
#
#         generation_config = GenerationConfig(
#             num_beams=5,
#             num_return_sequences=5,
#             return_dict_in_generate=True,
#             output_scores=True
#         )
#
#         if args.do_sample:
#             generation_config.temperature = args.temperature
#             generation_config.top_p = args.top_p
#             generation_config.top_k = args.top_k
#
#         with torch.no_grad():
#             generation_output = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,
#                 max_new_tokens=args.max_new_tokens,
#                 pad_token_id=tokenizer.pad_token_id
#             )
#         sequences = generation_output.sequences  # shape: (batch * num_return_sequences, seq_len)
#
#         outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
#         generated_responses = [output[output.rfind('assistant\n'):].split('assistant\n')[-1].replace('\n', '').strip()
#                                for output in outputs]
#
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)
#
# outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
