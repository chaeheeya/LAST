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
from itertools import count

from openai import OpenAI
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig, TrainerCallback, TrainerState, TrainerControl,TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from peft.utils.other import prepare_model_for_kbit_training

from tqdm import tqdm
from datetime import datetime
from pytz import timezone

from prompt_template import EVAL_only_response

# -------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # GPT
    parser.add_argument('--gpt_model', type=str, default="gpt-4.1")
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
    parser.add_argument('--data_path', type=str, default="")
    parser.add_argument('--traindata_len', type=int, default=1000)
    parser.add_argument('--no_shuffle', action='store_true')


    # GRPO config
    parser.add_argument('--num_generations', type=int, default=8)
    parser.add_argument('--max_completion_length', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--scale_rewards', type=str, default="group")


    # reward
    parser.add_argument('--reward_coeff', type=str, default="0.25, 0.25, 0.25, 0.25")
    parser.add_argument('--eval_prompt', type=str)



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
    parser.add_argument('--log_name', type=str, default="log_name")

    args = parser.parse_args()

    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(__file__)
    elif sysChecker() == "Windows":
        args.home = ''
    print(args.home)

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
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
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



class StepSaveAndLogCallback(TrainerCallback):
    def __init__(self, args, save_steps=100):
        self.log_name = args.log_name
        self.saved_model_path = args.output_path
        self.no_save = getattr(args, "no_save", False)
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        step = state.global_step

        # 매 100스텝마다 저장
        if step > 0 and step % self.save_steps == 0:
            save_path = os.path.join(self.saved_model_path, f"{self.log_name}_S{step}")
            if not self.no_save:
                model.save_pretrained(save_path)
                print(f"[INFO] Step {step}: model saved at {save_path}")


# 디버깅용 ---------------------------------------------------------------------------------------------------------
# ==== advantage 통계 찍기 ======
class InspectAdvantagesCallback(TrainerCallback):
    def on_substep_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None: return
        inp = getattr(trainer, "_debug_last_inputs", None)
        if not inp: return
        adv = inp.get("advantages", None)
        if adv is None: return
        try:
            m = float(adv.mean().item())
            s = float(adv.std().item())
            print(f"[DBG] step {state.global_step} advantages: shape={tuple(adv.shape)} mean={m:.4f} std={s:.4f}")
        except Exception:
            pass
        
# ===== gradient 생기는지 확인하기 ======
class GradSumCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None: return
        total = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                total += float(p.grad.abs().sum().item())
        print(f"[grad_sum] step {state.global_step}: {total:.3e}")


# 데이터셋 ----------------------------------------------------------------------------------------------------------
instruction = """Pretend you are a conversational recommender system. 
Create a response that the system should provide."""

def dataset_processing(args, dataset, tokenizer, instruction, rank, world_size, train_only_resp=False):

    # dialog -> utterance 로 쪼개기
    for data in dataset:
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
        
        ###########################################
        dialog = data['DIALOG'][-6:]
        dialog.insert(0, {'role': 'system', 'content': instruction})
        context = dialog
        # original_context_len = len(
        #     tokenizer.apply_chat_template(context[:-1], tokenize=True, add_generation_prompt=True))
        formatted_context = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=True)
        data["prompt"] = formatted_context

    # dataset format 맞추기
    print("Dataset length: ", len(dataset))


    if args.no_shuffle:
        print(dataset[0])
        pass

    else:
        random.shuffle(dataset)

    return dataset


# reward function 모음 ------------------------------------------------------------------------------------------------

# ------ dialog 저장 --------
DIALOG_ID_GEN = count(1)

def make_reward_sum(args, log_file):
    
    # state = {"dialog_counter": 0}
    
    
    def reward_sum(prompts=None, completions=None, **kwargs):
        # print("Evaluating by GPT!")
        '''
        TRL이 호출하는 reward function. 각 completions에 대한 보상 리스트를 반환해야함.
        하나의 dialog에 대해 생성한 응답들 각각에 대한 평가 항목별 점수를 합하여 정규화한 점수들
        :param group_evaluations:
        :return: List[float]
        '''
        reward_coeff = [float(i.strip()) for i in args.reward_coeff.split(',')]
        # print(f'reward coeff: {reward_coeff}')
        
        item_evaluations = []
        for topic, resp in zip(kwargs['TOPIC'], completions):
            
            pattern = r'\(\d+\)'
            match = re.search(pattern, topic)
            name = topic[:match.start()].strip()

            if name in resp:
                item_evaluations.append(1.0)
            else:
                item_evaluations.append(0.0)
        group_evaluations, dialogs = gpt_eval(args, prompts, completions)
        
        rewards = []
        for d, i in zip(group_evaluations, item_evaluations):
            s = [(float(d[k]) - 1.0) / (5.0 - 1.0)for k in ["informativeness", "fluency", "relevance"]]
            s.append(i)
            # min-max normalization
            r = sum([reward_coeff[i] * s[i] for i in range(len(reward_coeff))])
            rewards.append(r)
            
            
        #-------- txt 파일로 로그 저장하기 -------
        metrics = ["informativeness", "fluency", "relevance", "accuracy"]
        B = len(prompts)//args.num_generations if prompts is not None else 0
        G = int(args.num_generations)
        idx = 0
        
        for b in range(B):
            dialog_id = next(DIALOG_ID_GEN)
            start = idx
            end = min(start+G, len(completions))
            
            log_file.write(f"\n====================== Dialog {dialog_id} ======================\n")
            log_file.write("[Dialog]\n")
            log_file.write((dialogs[b*args.num_generations] or "") + "\n\n")
            
            for k, j in enumerate(range(start, end), start=1):
                log_file.write("----------------------------------------\n")
                log_file.write(f"[Completion {k}]\n")
                log_file.write((f"System: {completions[j]}" or "") + "\n")
                
                group_evaluations[j]['accuracy'] = item_evaluations[j]
                temp = {}
                for k in metrics:
                    temp[k] = group_evaluations[j][k]
                log_file.write("### reward by metric: " + json.dumps(temp, ensure_ascii=False) + "\n")
                log_file.write(f"### sum reward: {rewards[j]:.6f}\n\n")
            log_file.flush()
            idx = end
        
        # log_file.write(f'\n' + '='*30 + 'Dialog: {i}'+'\n')
        # for (d, c, r_val, r_sum) in zip(dialogs, completions, group_evaluations, rewards):
        #     log_file.write(f"Dialog:\n{d}\n")
        #     log_file.write(f"Completion:\nSystem: {c}\n\n")
        #     log_file.write(f"###GPT Reward: {json.dump(r_val}\n")
        #     log_file.write(f"###SUM Reward: {r_sum}\n")
        #     log_file.write("-"*80 + "\n")
        #     i +=1
        # log_file.flush()
        return rewards
    reward_sum.__name__="reward_sum"
    return reward_sum


def make_dummy_reward_sum(args):
    def dummy_reward(prompts=None, completions=None, **kwargs):
        # B*G 길이의 스칼라 리스트 보장
        return [1.0] * (args.batch_size * args.num_generations)
    dummy_reward.__name__="reward_sum"
    return dummy_reward



# GPT Eval 호출 ------------------------------------------------------------------------------------------------------
def gpt_eval(args, prompts: List[str], completions: List[str]):

    client = OpenAI(api_key=args.access_token)

    REQUIRED_KEYS = {"informativeness", "fluency", "relevance"}
    NEUTRAL = {"informativeness": 3, "fluency": 3, "relevance": 3}  # 항상 반환 강제 시 사용할 중립 점수
    MAX_RETRIES = 5
    BASE_BACKOFF = 0.6
    JITTER = 0.2

    EVAL_PROMPT = """I will provide you with a dialog and a response generated by a Conversational Recommender System (CRS).

Dialog:
%s

Response:
%s

Evaluate the response along explanation quality.
1) Informativeness: Does the explanation incorporate rich and meaningful knowledge about the recommended item?
2) Fluency: Is the explanation natural, coherent, and expressed with varied wording?
3) Relevance: Does the explanation highlight the features of the recommended item that are directly relevant to the dialog context?

Scoring: Use a 1–5 scale for each criterion.
- 1 point: Very poor. Fails almost entirely to meet the criterion.
- 2 points: Weak. Shows partial adequacy but remains insufficient.
- 3 points: Moderate. Meets the minimum requirement but lacks depth or strength.
- 4 points: Good. Clear, specific, and contextually appropriate, though not outstanding.
- 5 points: Excellent. Rich, highly natural, and strongly aligned with the context. Award only if it clearly stands out.

Output format:
<think>reasoning process here</think>
<answer>{"informativeness": <1–5>, "fluency": <1–5>, "relevance": <1–5>}</answer>"""


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
    
    def _convert_special_to_dialog(text:str) -> str:
        """
        <|start_header_id|>user<|end_header_id|> ... <|eot_id|> 
        같은 형식을 User: ... / System: ... 형태로 변환
        """
        
        # 매핑 규칙
        role_map = {
            "user": "User",
            "assistant": "System",
            "system": "System"
        }
        
        parts = text.split("<|start_header_id|>")
        dialogs = []

        for part in parts[2:]:  # 첫 번째는 <|begin_of_text|> 앞부분이라 skip
            try:
                role, content = part.split("<|end_header_id|>", 1)
                role = role.strip()
                # eot 기준으로 발화 내용 분리
                content = content.split("<|eot_id|>")[0].strip()
                if content:
                    dialogs.append(f"{role_map.get(role, role)}: {content}")
            except Exception as e:
                continue

        return "\n".join(dialogs)
        
        

    dialogs, results = [], []
    for dialog, response in zip(prompts, completions):
        
        dialog = _convert_special_to_dialog(dialog)
        dialogs.append(dialog)
        if not response.startswith('System: '):
            response = f'System: {response}'
        
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
    return results, dialogs



if __name__=="__main__":
    args = parse_args()

    # 토크나이저/모델/디바이스 --------------------------------------------------------------------------------------------

    ## 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        print('Set a pad token as <|pad|> in the tokenizer')
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    #     tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    ## 학습 대상 모델
    base_model = load_base_model(args.base_model)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    # base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    
    if args.model_path:
        model = PeftModel.from_pretrained(base_model, args.model_path, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=64, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj","v_proj"], task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(base_model, lora_cfg)
    
    # with torch.no_grad():
    #     model.get_input_embeddings().weight.data = model.get_input_embeddings().weight.data.to(torch.float32)
    #     model.get_output_embeddings().weight.data = model.get_output_embeddings().weight.data.to(torch.float32)

    # model = model.to(dtype=torch.bfloat16)
    # model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train}")
    assert n_train > 0
    model.print_trainable_parameters()
    
    print("[D] model param dtypes:", {p.dtype for p in model.parameters() if p.requires_grad})
    print("[D] head dtype:", model.get_output_embeddings().weight.dtype, 
      "embed dtype:", model.get_input_embeddings().weight.dtype)
    
    # model = load_base_model(args.base_model)
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id
    
    # model = prepare_model_for_kbit_training(model)   # input grad, layernorm 등 dtype정리
    # model.gradient_checkpointing_enable()           # 메모리 절약용(선택)
    # model.enable_input_require_grads()              # 반드시 필요
    # model.config.use_cache = False  
    
    # lora_cfg = LoraConfig(
    #     r=64, lora_alpha=16,
    #     target_modules=["q_proj", "v_proj"],      # 모델에 맞게
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type=TaskType.CAUSAL_LM,
    # )
    # model = get_peft_model(model, lora_cfg)
    # model.train()
    # model.print_trainable_parameters()     
    
    
    # model = model.to(dtype=torch.float16)
    # model.get_input_embeddings().to(torch.float16)
    # model.get_output_embeddings().to(torch.float16)
    # for _, p in model.named_parameters():
    #     if p.dtype == torch.float32:
    #         p.data = p.data.to(torch.float16)
    # for _, b in model.named_buffers():
    #     if hasattr(b, "dtype") and b.dtype == torch.float32:
    #         b.data = b.data.to(torch.float16)


    ## reference 모델 설정 - 안씀
    # ref_model = copy.deepcopy(model)
    # for p in ref_model.parameters():
    #     p.requires_grad = False
    # ref_model.eval()

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

    dataset = dataset_processing(args, train_dataset, tokenizer, instruction, rank, world_size)
    hf_train_dataset = HFDataset.from_list(dataset[:args.traindata_len])

    print('Dataset size:', len(hf_train_dataset))


    # 모델 저장 ---------------------------------------------------------------------------------------------------
    mdhm = datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S')
    args.output_path = os.path.join(args.home, 'GRPO_model', f'{mdhm}')
    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)
    print('Model saving path: %s' % args.output_path)

    # args.log_path = os.path.join(args.home, 'GRPO_log', f'{mdhm}')
    # if not os.path.isdir(args.log_path):
    #     os.mkdir(args.log_path)
    # print('Model logging path: %s' % args.log_path)

    # 학습 log 저장 ------------------------------------------------------------------------------------------------
    log_path = os.path.join(args.output_path, f"GRPO_{mdhm}_train_log.txt")
    log_file = open(log_path, 'a', buffering=1, encoding='UTF-8')


    
    # reward function 설정하기 -------------------------------------------------------------------------------------------
    reward_fn = make_reward_sum(args, log_file)
    reward_coeff = [float(i.strip()) for i in args.reward_coeff.split(',')]
    print(f'reward coeff: {reward_coeff}')


    # reward_fn = make_dummy_reward_sum(args)
    
    # # GRPO 설정 및 트레이너 ------------------------------------------------------------------------------------------------
    # training_args = GRPOConfig(
    #     output_dir=args.output_path,
    #     per_device_train_batch_size=args.batch_size,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     num_train_epochs=args.num_train_epochs,
    #     learning_rate=args.learning_rate,
    #     logging_steps=args.logging_steps,
    #     # bf16=False, fp16=False,
        
    #     # GRPO 핵심 파라미터
    #     # generation_batch_size=args.batch_size * args.num_generations,
    #     num_generations=args.num_generations, # 하나의 input당 생성하는 응답 개수
    #     max_completion_length=args.max_completion_length,
    #     temperature=args.temperature,
    #     #   top_p=args.top_p,
    #     beta=args.beta, # ref 모델과의 KL에 적용되는 파라미터
    #     scale_rewards=args.scale_rewards
    # )

    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        output_dir=args.output_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size = args.batch_size,
        remove_unused_columns=False, # to access the solution column in accuracy_reward
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        logging_strategy="steps",  # 스텝 단위 로깅
        logging_steps=100,         # 매 100스텝마다 log() 자동 호출
        save_strategy="no",        # (원하면 따로 설정)
        # Parameters that control de data preprocessing
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations, # 하나의 input당 생성하는 응답 개수
        max_prompt_length=512, # default: 512
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=hf_train_dataset,
        processing_class=tokenizer,
        callbacks=[StepSaveAndLogCallback(args, save_steps=100)],
    )


    # print("[DEBUG] N =", len(dataset))
    # print("[DEBUG] batch_size =", args.batch_size)
    # print("[DEBUG] epochs =", args.num_train_epochs)
    # print("[DEBUG] grad_accum =", args.gradient_accumulation_steps)

    # dl = DataLoader(hf_train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # print("[DEBUG] batches_per_epoch =", len(dl))
    # expected = math.ceil(len(hf_train_dataset) / args.batch_size) * args.num_train_epochs // max(1, args.gradient_accumulation_steps)
    # print("[DEBUG] expected_global_steps =", expected)

    
    print("🚀 GRPO 학습 시작")
    trainer.train()
    print("✅ Trainer.train() finished")
