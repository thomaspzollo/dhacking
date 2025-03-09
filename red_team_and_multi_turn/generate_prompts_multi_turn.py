import json
import random 
import os
os.environ['TRITON_CACHE_DIR'] = '/ /triton/'
os.environ['HF_DATASETS_CACHE'] = '/ /datasets/'
import torch
import gc 
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams 
from vllm.distributed.parallel_state import destroy_model_parallel
from datasets import load_dataset
import numpy as np


vllm_seed = 0
print("vllm_seed", vllm_seed)
# deterministic behaviour 
#np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
#torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 
device = "cuda" if torch.cuda.is_available() else "cpu"

domain_lms = {
'Gemma-2-9b-it' : {'instruction_tuned': True, 'model_path': '/ /weights/gemma-2-9b-it/', 
'max_new_tokens': 150, 'temperature': 1.2, 'top_p': 0.8, 'top_k': 40, 'repetition_penalty': 1.3
}
}

# for instruction tuned models
def gen_prompt_for_it(tokenizer, target_model_type, text):
        if 'llama' in target_model_type or 'Llama' in target_model_type: 
             messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
            ]
        else: 
             messages = [
            {"role": "user", "content": text},
            ]
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,

        )
        return prompt

def get_domain_prompt(instruction_tuned, tokenizer, target_model_type, domain_question): 
    if instruction_tuned: 
        text = f"""Please answer the following question: {domain_question}"""
        prompt = gen_prompt_for_it(tokenizer, target_model_type, text)
    
    return prompt 

def get_questions_from_dataset(domain):
    if domain == 'education': 
        ds = load_dataset("/ /datasets/gsm8k", "main")
        test_ds = ds['train']
        questions = [t_e['question'] for t_e in test_ds]
        return questions 
    
    if domain == 'health': 
        ds = load_dataset("/ /nikita/datasets/MedQuad-MedicalQnADataset")
        train_ds = ds['train']
        questions = train_ds['Question']
        return questions


def save_multi_turn_domain_answers_from_domain_model(domain, multi_turn_q_answers, domain_model_type): # using same question idx
    file_path = f'/ /nikita/dhacking/red_team_and_multi_turn/{domain}_answers/domain_prompts.json'
    with open(file_path, 'w') as json_file:
        json.dump(multi_turn_q_answers, json_file, indent=4)  # The 'indent' parameter is optional for pretty printing

def get_domain_answer_files(domain): 
    folder_path = f"/ /nikita/dhacking/red_team_and_multi_turn/{domain}_answers/"
    files = os.listdir(folder_path)  # Lists all files and directories
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]  # Lists only files
    red_lm_type_target_model_type_files = [f.split(".json")[0] for f in files]

    return set(red_lm_type_target_model_type_files)

def get_answers_to_domains_questions(): 
    all_domains_domain_lm_answers_to_domain_qs = {}
    for domain in ['education', 'health']: 
        existing_domain_answers = get_domain_answer_files(domain)
        all_domains_domain_lm_answers_to_domain_qs[domain] = {}
        for domain_lm_model_type in domain_lms.keys(): 
            model_path = domain_lms[domain_lm_model_type]['model_path']
            print('model_path', model_path)
            vllm_model = LLM(model=model_path, seed=vllm_seed) # use 2 devices. 
            tokenizer = vllm_model.get_tokenizer()
            sampling_params = SamplingParams(  
                max_tokens=domain_lms[domain_lm_model_type]['max_new_tokens'], 
                temperature=domain_lms[domain_lm_model_type]['temperature'], 
                top_p=domain_lms[domain_lm_model_type]['top_p'], 
                top_k=domain_lms[domain_lm_model_type]['top_k'], 
                repetition_penalty=domain_lms[domain_lm_model_type]['repetition_penalty']
                ) 
            check_if_exists = f"{domain_lm_model_type}"
            if check_if_exists in existing_domain_answers: 
                print("exists", check_if_exists)
                continue 
            domain_questions = get_questions_from_dataset(domain)
            prompts = []
            map_prompt_to_q_idx = {}
            domain_lm_answers_to_domain_qs = {}
            q_idx = 0
            for question in domain_questions: 
                if q_idx >= 5000: 
                    break
                prompt =  get_domain_prompt(domain_lms[domain_lm_model_type]['instruction_tuned'], tokenizer, domain_lm_model_type, question)
                if prompt not in map_prompt_to_q_idx: 
                    prompts.append(prompt)
                    map_prompt_to_q_idx[prompt] = q_idx
                else: 
                    continue # don't repeat the same prompt. 
                    #print("regroup_question", re_group_question)
                    #answers_red_lm_target_lm[q_idx][g] =  get_response_from_prompt(prompt, target_model, target_lm_model_type)
                    #print("answers_red_lm_target_lm[q_idx][g]: ", answers_red_lm_target_lm[q_idx][g])
                q_idx+=1
            vllm_responses = vllm_model.generate(prompts, sampling_params) 
            # get the groups of the regroup question from the response 
            print("vllm len", len(vllm_responses))
            for r in vllm_responses: 
                vllm_prompt = r.prompt
                q_idx =  map_prompt_to_q_idx[vllm_prompt]
                gen_text = r.outputs[0].text
                domain_lm_answers_to_domain_qs[q_idx] = f"{vllm_prompt} A: {gen_text}"         

            assert len(list(domain_lm_answers_to_domain_qs.keys())) == 5000 
            all_domains_domain_lm_answers_to_domain_qs[domain][domain_lm_model_type] = domain_lm_answers_to_domain_qs

            destroy_model_parallel()
            del vllm_model.llm_engine.model_executor.driver_worker
            del vllm_model # offload the target model that's completed. 
            gc.collect()
            torch.cuda.empty_cache()
    
    return all_domains_domain_lm_answers_to_domain_qs 

def concatenate_domain_q_ans_to_prompt(turn_prompts): 
    c_d_q_ans_to_prompt = ' '.join(turn_prompts)
    return c_d_q_ans_to_prompt

def save_multi_turn_domain_q_response_formatted(all_domains_answers_to_domain_qs): 
    multi_turn_all_domains_answers_to_domain_qs = {}
    for domain, answers_to_domain_qs in all_domains_answers_to_domain_qs.items(): 
        multi_turn_all_domains_answers_to_domain_qs[domain] = {}
        domain_lm_type = 'Gemma-2-9b-it'
        per_domain_lm_turn_answers_domain_qs = {}
        for turn in [2, 4, 6]: 
            turn_prompts = []
            turn_idx = 0
            per_domain_lm_turn_answers_domain_qs[turn] = {}
            answers_to_domain_target_lm_qs = all_domains_answers_to_domain_qs[domain][domain_lm_type]
            print("q_idxs", answers_to_domain_target_lm_qs.keys())
            for q_idx in range(0, 5000):
                d_q_answer = answers_to_domain_target_lm_qs[q_idx] 
                if turn_idx >= 1000: # 1000 prompts for each type of turn.
                    break
                turn_prompts.append(d_q_answer) # make sure to add the 0th q_idx one too for concat. 
                if (int(q_idx)+1) % (turn-1) == 0: 
                    print("turn", turn)
                    print("q_idx", q_idx)
                    print("len(t_prompts)", len(turn_prompts))
                    assert len(turn_prompts) == (turn - 1)
                    c_d_q_ans_to_prompt = concatenate_domain_q_ans_to_prompt(turn_prompts)
                    per_domain_lm_turn_answers_domain_qs[turn][turn_idx] = c_d_q_ans_to_prompt
                    turn_idx+=1
                    turn_prompts = [] # restart 

        multi_turn_all_domains_answers_to_domain_qs[domain][domain_lm_type] = per_domain_lm_turn_answers_domain_qs
        save_multi_turn_domain_answers_from_domain_model(domain, multi_turn_all_domains_answers_to_domain_qs[domain][domain_lm_type], domain_lm_type)
    
    return multi_turn_all_domains_answers_to_domain_qs
            

all_domains_answers_to_domain_qs = get_answers_to_domains_questions()
#print(all_domains_answers_to_domain_qs)
multi_turn_all_domains_answers_to_domain_qs = save_multi_turn_domain_q_response_formatted(all_domains_answers_to_domain_qs)