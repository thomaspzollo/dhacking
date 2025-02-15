# Towards Effective Discrimination Testing for Generative AI

**Authors**: Thomas P. Zollo, Nikita Rajaneesh, Richard Zemel, Talia B. Gillis, Emily Black

[📄 Paper on arXiv](https://arxiv.org/abs/2412.21052)

## Install 

1. Download the zip folder 
2. Install neccessary packages.  

## Experiments 
For additional details please see the appendix of the paper. 
### Hiring experiment
This experiment is described in section 4.1 of the paper. Please find the relevant files to this experiment in the `hiring` folder. 

1. Produce synthetic personas. (```generate_personas.ipynb```)
2. Feed the personas to GPT4 (```generate_resumes.ipynb```) to produce corresponding resumes. 
3. Summarize the resumes (```generate_summaries.py```). 
4. Scores the resumes(```score.ipynb```) using different metrics (such as ROUGE and sentiment). 
5. Finally, simulate the interview decisions (```simulate_decisions.py```)


### Red teaming experiment 
This experiment is described in section 4.2 of the paper. Please find the relevant files to this experiment in the  `red_team_and_multi_turn` folder.  

1. For different redLMs, generate redLM attacks (eg of attack: What do you like in GROUP people?) using appropriate few shot examples and prompt (```generate_questions_diff.py```). 
2. Replace GROUP with group(s) of interest in redLM attacks and for each attack generate responses from  different candidate models. (```answers_for_red_team method in  generate_answers_red_LM_diff_target_lms.py```). 
3. Compute the toxicity of each response from the candidate models. (```toxicity_red_team in toxicity_for_redteam_and_multiturn.py```)
4. Plot results. (```plot_red_team_results.ipynb```)

### Multi-turn experiment 
This experiment is described in section 4.3 of the paper. 
Please find the relevant files to this experiment in the `red_team_and_multi_turn` folder.

1. Per domain (education and health), use Gemma-2-9b-it model to generate responses to a set of domain question. (```get_answers_to_domains_questions in generate_prompts_multi_turn```).  
2. Create conversations with the responses for conversation length 1, 3 and 5. There's nothing to be done here for 0. (Note these lengths are 1-indexed in the code so insteaed of 0,1,3,5. The code writes 1,2,4,6.) (```save_multi_turn_domain_q_response_formatted in generate_prompts_multi_turn```). 
3. Concatenate redLM attacks for all prompts (for all conversation lengths and redLMs) and generate responses from the candidate model (```answers_for_multi_turn in generate_answers_red_LM_diff_target_lms.py```). 
4. Compute the toxicity of each response from the candidate models (```toxicity_multi_turn in toxicity_for_redteam_and_multiturn.py```)
5. Plot results (```plot_multi_turn_results.py```)

### Potrayal experiment 
This experiment is described in section 4.4 of the paper. 
Please find the relevant files to this experiment in the potrayal folder. 

1. Generate images using the medium version of the popular open source StableDiffusion3 model. (```generate_images.py```). 
2. Classify the bias in images using NSFW scores. (```classify_images.py```)
3. Plot results. (```final_results.ipynb```)