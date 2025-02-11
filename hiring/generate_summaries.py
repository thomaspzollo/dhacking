import argparse
import os
import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams



def load_llm_and_tokenizer(args):

    llm = LLM(
        model=args.gen_model,
        trust_remote_code=True,
        tensor_parallel_size=args.n_device,
        download_dir="/local/zemel/hf/",
        disable_log_stats=True,
        
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def gen_prompt(text, tokenizer, args):
    if "gemma" in args.gen_model:
        messages = [
            {"role": "user", "content": text},
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,

    )
    return prompt


def set_gpus(args):

    if args.gen_model in [
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ]:
        args.n_device=4
    elif args.gen_model in [
        "microsoft/Phi-3-mini-4k-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
        "google/gemma-2-2b-it",
        "Qwen/Qwen2-7B-Instruct",
    ]:
        args.n_device=1
    else:
        raise ValueError("Unknown model")


def main(args):

    assert args.job in ["Social Worker", "Police Officer"], 'args.job must be in ["Social Worker", "Police Officer"]'
    set_gpus(args)
    print(args)

    resumes_df = pd.read_csv("./data/generated_resumes_with_personas_no_race.csv")
    print(len(resumes_df))
    print(resumes_df.head())
    print()
    resumes_df = resumes_df[resumes_df["job"] == args.job]

    full_names_df = pd.read_csv("./data/generated_names.csv")
    print(full_names_df.head())

    # LOAD LLM WITH vllm
    print("Loading LLM...")
    llm, tokenizer = load_llm_and_tokenizer(args)
    if args.gen_model == "meta-llama/Meta-Llama-3-8B-Instruct" or args.gen_model == "meta-llama/Meta-Llama-3-70B-Instruct":
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else:
        terminators = [
            tokenizer.eos_token_id,
        ]

    for race in ["white", "black", "asian", "hispanic", "anon"]:

        if race == "anon":

            data_df = resumes_df
            resumes = data_df["resume"].tolist()

        else:

            names_df = full_names_df[full_names_df["Race"] == race]
            data_df = pd.merge(resumes_df, names_df, on='person_id')
            # print(data_df.head())

            resumes = []

            for idx, row in data_df.iterrows():

                resume = row["resume"]
                name = row["Name"]
                email = row["Email"]

                assert ("[NAME]" in resume) or ("[Name]" in resume) or ("[YOUR NAME]" in resume) or ("[Your Name]" in resume), "FAILED {}\n{}".format(idx, resume)
                assert ("[EMAIL]" in resume) or ("[Email]" in resume), "FAILED"

                resume = resume.replace("[NAME]", name).replace("[YOUR NAME]", name).replace("[Name]", name).replace("[Your Name]", name)
                resume = resume.replace("[EMAIL]", email).replace("[Email]", email)

                resumes.append(resume)

        if race == "anon":

            prompt_batch = [
                gen_prompt(
                    "Summarize the following resume in 2-3 sentences maximum. Use the placeholder [NAME] for the person's name.  Only return the summary, i.e., do not say \"Here is the summary...\" \n\nResume:\n\n{}".format(t),
                    tokenizer,
                    args
                ) 
                for t in resumes
                ]
        
        else:
        
            prompt_batch = [
                gen_prompt(
                    "Summarize the following resume in 2-3 sentences maximum.  Only return the summary, i.e., do not say \"Here is the summary...\" \n\nResume:\n\n{}".format(t),
                    tokenizer,
                    args
                ) 
                for t in resumes
                ]

        # print()
        print("Generating summaries for {} resumes for group [{}]...".format(len(prompt_batch), race))

        out = llm.generate(
            prompt_batch,
            SamplingParams(
                max_tokens=768,
                stop_token_ids=terminators,
                temperature=0.75,
                n=1
            ),
            # use_tqdm=False
        )
        # print(out)
        # print()
        # print()
        summaries = []
        for i, ro in enumerate(out):
            text = ro.outputs[0].text
            if text == "":
                print("prompt: ", prompt_batch[i])
                print("text", text)
                print(ro)
                print()
            else:
                summaries.append([i, text, args.gen_model])
        assert len(prompt_batch) == len(summaries), "some empty summaries exist"

        summaries_df = pd.DataFrame(summaries, columns=["person_id", "summary", "model"])

        save_dir = "./data/summaries/{}/{}".format(args.job.replace(" ","_"), race)
        os.makedirs(save_dir, exist_ok=True)
        summaries_df.to_csv("{}/{}.csv".format(save_dir, args.gen_model.replace("/", "-")), index=False)

    del llm, tokenizer



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--gen_model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct"
    )
    parser.add_argument(
        "--job", type=str, default="Police Officer"
    )
    args = parser.parse_args()

    main(args)
