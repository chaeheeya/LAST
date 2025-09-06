import os
import sys
import json
import logging
import argparse
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainerState, TrainerControl, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from datetime import datetime
from pytz import timezone

# === Hyperparameters ===
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# TOTAL_EPOCHS = 5
# LEARNING_RATE = 3e-5
# BATCH_SIZE = 4
# LOGGING_STEPS = 100


instruction = """Pretend you are a conversational recommender system. 
Create a response that the system should provide."""


class QueryEvalCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.saved_model_path = output_dir

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        wrapper_model = kwargs['model']  # 전체 wrapper 모델
        peft_model = wrapper_model.model  # PEFT 모델 내부
        epoch = state.epoch
        path = os.path.join(self.saved_model_path, f'E{round(epoch)}')

        # 1. PEFT LoRA 파라미터만 저장
        peft_model.save_pretrained(path)
        # 3. config도 같이 저장
        peft_model.config.save_pretrained(path)
        print(f"Epoch {state.epoch} finished, saving model to {self.saved_model_path}")


class Dataset_processing(Dataset):
    def __init__(self, args, json_dataset, tokenizer, instruction, rank, world_size, train_only_resp=False, ):
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
            response_utt = data['RESPONSE'].split('System: ')[1].strip()
            context.append({'role': "assistant", 'content': response_utt})
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
            original_context_len = len(
                tokenizer.apply_chat_template(context[:-1], tokenize=True, add_generation_prompt=True))
            formatted_context = self.tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=False)
            # formatted_context = self.tokenizer(formatted_context, padding='max_length', truncation=True, max_length=1024, return_tensors='pt')
            # dialog_text = "\n".join([f"{i['role']}: {i['content']}" for i in data['dialog']])

            tokenized_context = tokenizer(formatted_context, truncation=True, add_special_tokens=False)

            input_ids = tokenized_context.input_ids
            labels = input_ids.copy()

            if train_only_resp:
                labels = [token if idx >= original_context_len else -100 for idx, token in enumerate(input_ids)]

            self.formatted_dataset.append({'input_ids': input_ids, "labels": labels})

        # self.tokenizer.apply_chat_template([{'role': 'system', 'content': instruction}] + inspired2_train[0]['dialog'], tokenize=True, padding=True, max_length=128, add_generation_prompt=True)

    def __len__(self):
        # 데이터 샘플 개수를 반환
        return len(self.formatted_dataset)

    def __getitem__(self, idx):
        data = self.formatted_dataset[idx]
        return {'input_ids': data['input_ids'], "labels": data['labels']}


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', type=str, default="train")
    parser.add_argument('--batch_size', type=int, default=2)

    # Generation
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)

    # Train
    parser.add_argument('--train_data', type=str, default="", help="Write only the data name(not the path)")
    parser.add_argument('--deepspeed', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--step_size', type=int, default=300)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--train_only_response', action='store_true')

    parser.add_argument('--lora_weights', type=str, default=None)
    parser.add_argument('--access_token', type=str, default="")
    parser.add_argument('--cnt', type=int, default=0)
    parser.add_argument('--log_name', type=str, default="")

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


def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"
    return tokenizer


# def prepare_dataset(data_path, tokenizer, rank, world_size, train_only_interaction=False):
#     all_data = json.load(open(data_path, 'r', encoding='utf-8'))
#     # all_data = all_data[:10]
#     shuffle(all_data)
#
#     # 데이터 분산 처리
#     data = all_data[rank::world_size]
#
#     dataset = []
#
#     for example in data:
#
#         dialog = example['dialog'][-5:]
#         dialog.insert(0, {'role': 'system', 'content': instruction})
#         interaction = example['interaction'][:-1]
#         # context = dialog + interaction
#
#         context = dialog + interaction
#
#         original_context_len = len(tokenizer.apply_chat_template(dialog, tokenize=True, add_generation_prompt=True))
#         prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=False)
#         # if prompt in dataset:
#         #     continue
#
#         # tokenized_prompt = tokenizer(prompt, truncation=True, max_length=512, add_special_tokens=False)
#         tokenized_prompt = tokenizer(prompt, truncation=True, add_special_tokens=False)
#         input_ids = tokenized_prompt.input_ids
#         labels = input_ids.copy()
#         if train_only_interaction:
#             labels = [token if idx >= original_context_len else -100 for idx, token in enumerate(input_ids)]
#
#         dataset.append({'input_ids': input_ids, "labels": labels})
#
#     return dataset


def main(args):
    tokenizer = setup_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
    base_model = load_base_model("meta-llama/Llama-3.1-8B-Instruct")
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = load_peft_model(base_model, args.model_path)

    # model = get_peft_model(base_model, lora_config)

    # wandb.init(
    #     project="learning2interact",  # 원하는 wandb 프로젝트 이름
    #     name=args.log_name,           # 실험 run 이름
    # )

    # LoRA Configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    rank, world_size = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # Prepare dataset
    #    data_path = os.path.join(args.home, 'data', 'redial_processed_train_sft_gpt_turn3.json')   # BS수정
    dataset_path = os.path.join(args.home, 'dataset', 'sft_train', args.train_data)
    dataset = json.load(open(dataset_path, 'r', encoding='utf-8'))

    train_dataset = Dataset_processing(args, dataset, tokenizer, instruction, rank, world_size, train_only_resp=args.train_only_response)
    # dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Logging 설정
    mdhm = datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S')
    dir_name = ''
    if 'benchmark' in dataset_path:
        dir_name = 'benchmark'
    elif 'GPT' in dataset_path:
        if 'refined' in dataset_path:
            dir_name = 'refined_gpt'
        else:
            dir_name = 'gpt'

    # log_name = args.log_name
    #
    #
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(message)s",
    #     handlers=[
    #         logging.FileHandler(log_file, encoding="utf-8"),
    #         logging.StreamHandler(sys.stdout)
    #     ]
    # )

    model_path = os.path.join(args.home, 'model_weights', f'{dir_name}', f'{mdhm}')
    print('Model saving path: %s' % model_path)

    training_args = TrainingArguments(
        deepspeed=args.deepspeed if args.deepspeed != '' else None,
        output_dir=model_path,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy='no',
        bf16=True,
        fp16=False,  # fp16,
        remove_unused_columns=False,
        # report_to='wandb'
        # logging_dir="./logs",
        # report_to="wandb" if args.use_wandb else "none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[QueryEvalCallback(training_args.output_dir)]
    )

    # 학습 시작
    logging.info("🚀 SFT 학습 시작")
    trainer.train()
    logging.info("✅ Trainer.train() finished")

    # 모델 저장
    # LoRA adapter 저장

    # if trainer.accelerator.is_main_process:
    # print(int(os.environ.get("LOCAL_RANK", 0)))
    # if trainer.is_world_process_zero():
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    print(f"[RANK {local_rank}] Saving model...", flush=True)
    # if local_rank == 0:
    print("➡ model.save_pretrained()", flush=True)
    model.save_pretrained(model_path)
    print("✅ model 저장됨", flush=True)

    print("➡ tokenizer.save_pretrained()", flush=True)
    tokenizer.save_pretrained(model_path)
    print("✅ tokenizer 저장됨", flush=True)

    # # 모델 merge 및 저장 (LoRA → base weights에 합치기)
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(model_path + "_merged")
    # tokenizer.save_pretrained(model_path + "_merged")
    # logging.info("✅ Merge된 모델 저장 완료")
    #ckckckckck

if __name__ == "__main__":
    args = parse_args()
    main(args)
