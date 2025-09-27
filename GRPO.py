import os
import json
import pickle
import copy
import math
import time
import random
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Any
from functools import partial

from openai import OpenAI
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig, TrainerCallback, TrainerState, TrainerControl,TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from tqdm import tqdm
from datetime import datetime
from pytz import timezone

# -------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # GPT
    parser.add_argument('--gpt_model', type=str, default="")
    parser.add_argument('--access_token', type=str, default="")

    # Train
    parser.add_argument('--base_model', type=str, default="") #LLaMA-3.1: "meta-llama/Llama-3.1-8B-Instruct"
    parser.add_argument('--model_path', type=str, default="", help="SFT model path")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=300)

    # GRPO config
    parser.add_argument('--num_generations', type=int, default=8)
    parser.add_argument('--max_completion_length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--scale_rewards', type=str, default="group")



    # # Generation
    # parser.add_argument('--max_new_tokens', type=int, default=100)
    # parser.add_argument('--num_beams', type=int, default=1)
    # parser.add_argument('--do_sample', action='store_true')
    # parser.add_argument('--temperature', type=float, default=1.0)
    # parser.add_argument('--top_k', type=int, default=50)

    # Train
    parser.add_argument('--deepspeed', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=-1)

    #
    # parser.add_argument('--access_token', type=str, default="")
    # parser.add_argument('--cnt', type=int, default=0)
    # parser.add_argument('--log_name', type=str, default="")

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


# 모델 load function ----------------------------------------------------------------------------------------------
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


# 모델 저장 -------------------------------------------------------------------------------------------------------
class SaveEveryEpochCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.saved_model_path = output_dir

    def on_epoch_end(self,args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        wrapper_model = kwargs['model']  # 전체 wrapper 모델
        peft_model = wrapper_model.model  # PEFT 모델 내부
        epoch = state.epoch
        path = os.path.join(self.saved_model_path, f'E{round(epoch)}')

        # 1. PEFT LoRA 파라미터만 저장
        peft_model.save_pretrained(path)
        # 3. config도 같이 저장
        peft_model.config.save_pretrained(path)
        print(f"Epoch {state.epoch} finished, saving model to {self.saved_model_path}")



# 데이터셋 ----------------------------------------------------------------------------------------------------------
instruction = """Pretend you are a conversational recommender system. 
Create a response that the system should provide."""

class Dataset_processing(Dataset):
    def __init__(self, args, json_dataset, tokenizer, instruction, rank, world_size, train_only_resp=False):
        self.args = args
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.dataset = json_dataset

        # dialog -> utterance 로 쪼개기
        for data in self.dataset:
            dialog = data['DIALOG'].split('\n')
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
            data['DIALOG'] = context

        # dataset format 맞추기
        print("Dataset length: ", len(self.dataset))

        random.shuffle(self.dataset)
        data = self.dataset[rank::world_size]

        self.formatted_dataset = []
        for data in self.dataset:
            dialog = data['DIALOG'][-6:]
            dialog.insert(0, {'role': 'system', 'content': self.instruction})

            context = dialog
            # original_context_len = len(
            #     tokenizer.apply_chat_template(context[:-1], tokenize=True, add_generation_prompt=True))
            formatted_context = self.tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
            self.formatted_dataset.append({"prompt": formatted_context})
            # formatted_context = self.tokenizer(formatted_context, padding='max_length', truncation=True, max_length=1024, return_tensors='pt')
            # dialog_text = "\n".join([f"{i['role']}: {i['content']}" for i in data['dialog']])

            # self.formatted_dataset.append({'input_ids': input_ids, "labels": labels})

        # self.tokenizer.apply_chat_template([{'role': 'system', 'content': instruction}] + inspired2_train[0]['dialog'], tokenize=True, padding=True, max_length=128, add_generation_prompt=True)

    def __len__(self):
        # 데이터 샘플 개수를 반환
        return len(self.formatted_dataset)

    def __getitem__(self, idx):
        return self.formatted_dataset[idx]




# reward function 모음 ------------------------------------------------------------------------------------------------
def reward_sum(args, prompts=None, completions=None, **kwargs):
    '''
    TRL이 호출하는 reward function. 각 completions에 대한 보상 리스트를 반환해야함.
    하나의 dialog에 대해 생성한 응답들 각각에 대한 평가 항목별 점수를 합하여 정규화한 점수들
    :param group_evaluations:
    :return: List[int]
    '''
    group_evaluations = gpt_eval(args, prompts, completions)
    rewards = []
    for d in group_evaluations:
        s = sum(float(d[k]) for k in ['informativeness', 'fluency', 'relevance', 'validity'])
        # min-max normalization
        r = (s-4.0) / 16.0
        r = max(0.0, min(1.0, r))
        rewards.append(r)
    return rewards



# GPT Eval 호출 ------------------------------------------------------------------------------------------------------
def gpt_eval(args, prompts: List[str], completions: List[str]):

    client = OpenAI(api_key=args.access_token)

    REQUIRED_KEYS = {"validity", "informativeness", "fluency", "relevance"}
    NEUTRAL = {"validity": 3, "informativeness": 3, "fluency": 3, "relevance": 3}  # 항상 반환 강제 시 사용할 중립 점수
    MAX_RETRIES = 5
    BASE_BACKOFF = 0.6
    JITTER = 0.2

    EVAL_PROMPT = """I will provide you with a dialog and a response generated by a Conversational Recommender System (CRS).

Dialog:
%s

Response:
%s

Evaluate the response along two dimensions: (A) Recommendation Quality and (B) Explanation Quality.

A. Recommendation Quality
1) Validity: Does the recommended item align with the user’s preferences expressed in the dialog?

B. Explanation Quality
1) Informativeness: Does the explanation incorporate rich and meaningful knowledge about the item?
2) Fluency: Is the explanation natural, coherent, and expressed with varied wording?
3) Relevance: Does the explanation highlight item features that are directly relevant to the dialog context?

Scoring: Use a 1–5 scale for each criterion.
- 1 point: Very poor. Fails almost entirely to meet the criterion.
- 2 points: Weak. Shows partial adequacy but remains insufficient.
- 3 points: Moderate. Meets the minimum requirement but lacks depth or strength.
- 4 points: Good. Clear, specific, and contextually appropriate, though not outstanding.
- 5 points: Excellent. Rich, highly natural, and strongly aligned with the context. Award only if it clearly stands out.

Output format:
<think>reasoning process here</think>
<answer>{"validity": <1–5>, "informativeness": <1–5>, "fluency": <1–5>, "relevance": <1–5>}</answer>"""


    def _extract_between(text: str, start_tag="<answer>", end_tag="</answer>") -> str:
        s = text.find(start_tag)
        if s == -1:
            raise ValueError("missing <answer> tag")
        e = text.find(end_tag, s + len(start_tag))
        if e == -1:
            raise ValueError("missing </answer> tag")
        content = text[s + len(start_tag): e].strip()
        if not content:
            raise ValueError("empty content between answer tags")
        return content


    def _parse_and_validate_scores(json_str: str) -> dict:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("evaluation must be a JSON object")
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise KeyError(f"missing keys: {missing}")

        cleaned = {}
        for k in REQUIRED_KEYS:
            v = data[k]
            # 숫자/문자 섞여 들어오는 상황 방지: 정수로 보정
            try:
                v_int = int(round(float(v)))
            except Exception:
                raise ValueError(f"value for {k} not a number: {v}")
            if not (1 <= v_int <= 5):
                raise ValueError(f"value for {k} out of range [1,5]: {v_int}")
            cleaned[k] = v_int
        return cleaned

    results = []
    for dialog, response in zip(prompts, completions):
        instruction = EVAL_PROMPT % (dialog, response)
        eval_score = None

        for attempt in range(MAX_RETRIES+1):
            try:
                evaluation = client.chat.completions.create(
                    model=args.gpt_model,
                    messages=[{"role": "user", "content": instruction}],
                    temperature=0,
                )
                raw_text = evaluation.choices[0].message.content.strip()
                inside_tag_text = _extract_between(raw_text, "<answer>", "</answer>")
                eval_score = _parse_and_validate_scores(inside_tag_text)  # 모든 key가 존재하는지 검사
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    eval_score = NEUTRAL.copy()
                    break
                sleep_s = (BASE_BACKOFF * (2 ** attempt)) * (1.0 + random.uniform(-JITTER, JITTER))
                time.sleep(max(0.2, sleep_s))

        results.append(eval_score)
    return results



if __name__=="__main__":
    args = parse_args()

    # 토크나이저/모델/디바이스 --------------------------------------------------------------------------------------------

    ## 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        print('Set a pad token as <|pad|> in the tokenizer')
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    ## 학습 대상 모델
    model = load_base_model(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.model_path:
        model = load_peft_model(model, args.model_path)

    ## reference 모델 설정
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    ## 디바이스
    rank, world_size = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()


    # 데이터셋 -------------------------------------------------------------------------------------------------------
    if args.data_path != '':
        args.data_path = os.path.join(args.home, 'dataset', args.data_path)
    else:
        raise ValueError("No data path specified")

    if 'pkl' in args.data_path:
        train_dataset = pickle.load(open(args.data_path, 'rb'))
    elif 'json' in args.data_path:
        train_dataset = json.load(open(args.data_path, 'r', encoding='utf-8'))
    else:
        raise ValueError('Invalid data path')

    dataset = Dataset_processing(args, train_dataset, tokenizer, instruction, rank, world_size)
    hf_train_dataset = HFDataset.from_list(train_dataset)
    print('Dataset size:', len(dataset))


    # 모델 저장 ---------------------------------------------------------------------------------------------------
    mdhm = datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S')
    args.output_path = os.path.join(args.home, 'GRPO_model', f'{mdhm}')
    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)
    print('Model saving path: %s' % args.output_path)


    # 학습 log 저장 ------------------------------------------------------------------------------------------------
    log_path = os.path.join(args.output_path, "train_log.txt")
    log_file = open(log_path, 'a', buffering=1, encoding='UTF-8')


    # GRPO 설정 및 트레이너 ------------------------------------------------------------------------------------------------
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        # GRPO 핵심 파라미터
        num_generations=args.num_generations, # 하나의 input당 생성하는 응답 개수
        max_completion_length=args.max_completion_length,
        temperature=args.temperature, top_p=args.top_p,
        beta=args.beta, # ref 모델과의 KL에 적용되는 파라미터
        scale_rewards=args.scale_rewards,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=partial(reward_sum, args),
        args=training_args,
        train_dataset=hf_train_dataset,
        processing_class=tokenizer,
        callbacks=[SaveEveryEpochCallback(training_args.output_dir)],
    )

    print("🚀 GRPO 학습 시작")
    trainer.train()
    print("✅ Trainer.train() finished")
