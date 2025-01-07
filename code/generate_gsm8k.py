import path_init

from tqdm import tqdm
from statistics import mode
import argparse

import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
import os

from utils.answer_clean_utils import answer_cleansing
from collections import Counter
import os
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

ds_path_dict = {
    "GSM8K": "GSM8K/gsm8k_train",
    "MATH": "MATH/MATH_train",
    "GSM8K_rephrased": "GSM8K/gsm8k_train-cleaned_rephrased_questions",
    "MATH_rephrased": "MATH/MATH_train-cleaned_rephrased_questions",
}

# ---------------
MAX_INT = sys.maxsize

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def get_prompt():
    prompt_file = os.path.join("../configs/", "ansaug_cot_gsm8k.txt")
    with open(prompt_file, "r", encoding='utf-8') as f:
        prompt = f.read().strip()
    return prompt

def gsm8k_test(args, examples, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=2):
    model = args.model_name

    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    few_shot_prompt = get_prompt()
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request like given examples below:\n\n"
        "{}\n### Instruction:\n{{instruction}}\n\n### Response: Let's think step by step.".format(few_shot_prompt)
    )
    # print('promt =====', problem_prompt)

    for idx, item in enumerate(examples):
        temp_instr = problem_prompt.format(instruction=item["question"])
        gsm8k_ins.append(temp_instr)
        temp_ans = item['answer_detail'].split('#### ')[1]
        temp_ans = int(temp_ans.replace(',', ''))
        gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    examples = examples[start:end]
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    result = []
    res_completions = []
    for idx, (prompt, prompt_answer, e) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers, examples)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            extract(e, generated_text)

def extract(e, reply):
    e['pred_answer'] = reply
    e['pred_answer_cleaned'] = answer_cleansing(pred=reply, ds_name='gsm8k')
# ---------------------------

def get_right_data(args):
    iter = f"_{args.iter}" if len(args.iter) > 0 else ""
    # pre_file = f"../../../../../../../data/xiangchi/data/self/{ds_path_dict[args.ds]}_{args.method_name}_answer_qwen_{iter}.json"
    # out_file = f"../../../../../../../data/xiangchi/data/self/{ds_path_dict[args.ds]}_{args.method_name}_answer_qwen_right_{iter}.json"
    pre_file = f"/research/data/transfer/data/xiangchi/data/self/{ds_path_dict[args.ds]}_{args.method_name}_answer_qwen_{iter}.json"
    out_file = f"/research/data/transfer/data/xiangchi/data/self/{ds_path_dict[args.ds]}_{args.method_name}_answer_qwen_right_{iter}.json"
    with open(pre_file, 'r') as file:
        data = json.load(file)

    # Function to filter and rename keys
    def process_entries(entries):
        processed_entries = []
        for entry in entries:
            if entry['pred_answer_cleaned'] == entry['answer']:
                # Renaming 'answer' to 'query' and 'pred_answer' to 'response'
                entry['query'] = entry.pop('question')
                entry['response'] = entry.pop('pred_answer')
                processed_entries.append(entry)
        return processed_entries

    # Filter and rename the data
    filtered_renamed_data = process_entries(data)

    # Save the processed data back to a new JSON file
    with open(out_file, 'w') as file:
        json.dump(filtered_renamed_data, file, indent=4)

    print("Processed data has been saved.")

def batch_get_qwen_v2(examples, eng, pre_fun, post_fun,
                  logger=None, max_tokens=1024, temperature=0.0, timeout=20, max_try=0, **kwargs):
    prompts = [pre_fun(_) for _ in examples]

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = []
    for e in examples:
        prompt = pre_fun(e)
        messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    for i, e in enumerate(examples):
        reply = response[i]
        post_fun(e, reply)

def batch_get_qwen(examples, eng, pre_fun, post_fun,
                  logger=None, max_tokens=1024, temperature=0.0, timeout=20, max_try=0, **kwargs):
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for e in examples:
        prompt = pre_fun(e)
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        post_fun(e, response)

def batch_get_api_merge(examples, eng, pre_fun, post_fun, temperature=0.7, timeout=20, max_try=0, **kwargs):
    if eng in ("qwen"):
        batch_get_qwen(examples, eng, pre_fun=pre_fun, post_fun=post_fun,
                       temperature=temperature, timeout=timeout, max_try=max_try, **kwargs)



class ForwardReasoning():
    def __init__(self, args):
        self.args = args
        self.ds_name = args.ds
        self.temperature = args.temp
        self.iter = f"_{args.iter}" if len(args.iter) > 0 else ""

        self.eng = args.eng
        self.num_repeat = args.num_repeat
        self.method_name = self.get_method_name()


        self.json_file = f"{ds_path_dict[self.ds_name]}.json"
        # self.save_file = f"../../../../../../../data/xiangchi/data/self/{ds_path_dict[self.ds_name]}_{self.method_name}_answer_qwen_{self.iter}.json"
        # self.save_stat_file = f"../../../../../../../data/xiangchi/data/self/{ds_path_dict[self.ds_name]}_{self.method_name}_answer_qwen_{self.iter}_stat.json"
        self.save_file = f"/research/data/transfer/data/xiangchi/data/self/{ds_path_dict[self.ds_name]}_{self.method_name}_answer_qwen_{self.iter}.json"
        self.save_stat_file = f"/research/data/transfer/data/xiangchi/data/self/{ds_path_dict[self.ds_name]}_{self.method_name}_answer_qwen_{self.iter}_stat.json"
        if not args.cont:
            with open(self.json_file) as f:
                self.examples = json.load(f)
                self.examples = np.repeat(self.examples, self.num_repeat).tolist()
            self.save_data()

        with open(self.save_file) as f:
            self.examples = json.load(f)

        if "GSM8K" in self.ds_name:
            self.prompt = self.get_prompt("ansaug_cot_gsm8k.txt")
        elif "MATH" in self.ds_name:
            self.prompt = self.get_prompt("ansaug_cot_math.txt")
        else:
            raise ValueError(f"unknown dataset={self.ds_name}")

    def save_data(self):
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, ensure_ascii=False, indent=4)

    def get_method_name(self):
        return "SCComplexCoT"



    def get_prompt(self, prompt_file_name):
        prompt_file = os.path.join("../configs/", prompt_file_name)
        with open(prompt_file, "r", encoding='utf-8') as f:
            prompt = f.read().strip()
        return prompt

    def save_ans_stat(self):
        examples_collect = {}
        for e in self.examples[0:len(self.examples)]:
            question = e['question']
            if question not in examples_collect:
                examples_collect[question] = {}

                for k in ["answer", "question", "answer_detail"]:
                    examples_collect[question][k] = e[k]

                examples_collect[question]['pred_answer_cleaned_list'] = []

            examples_collect[question]['pred_answer_cleaned_list'].append(e['pred_answer_cleaned'])

        stat_list = []
        for e in examples_collect.values():
            counter = Counter(e['pred_answer_cleaned_list'])

            e["ans_stat"] = dict(counter)

            del e['pred_answer_cleaned_list']
            stat_list.append(e)

        with open(self.save_stat_file, 'w', encoding='utf-8') as f:
            json.dump(stat_list, f, ensure_ascii=False, indent=4)

    def evaluate(self, end_idx):
        result_stat_dict = {}
        for e in self.examples[0:end_idx]:
            question = e['question']

            if question not in result_stat_dict:
                result_stat_dict[question] = []

            result_stat_dict[question].append(e)

        num_correct = 0
        for q in result_stat_dict:
            e_list = result_stat_dict[q]
            answer = e_list[0]['answer']
            pred_answers = [_['pred_answer_cleaned'] for _ in e_list]
            freq_answer = mode(pred_answers)

            if freq_answer == answer:
                num_correct += 1
        msg = f"acc: {100 * num_correct / len(result_stat_dict.keys()):.4f}"
        print(msg)
        return num_correct, len(result_stat_dict.keys()), num_correct / len(result_stat_dict.keys())

    def fetch_data_from_openai(self):
        def wrap(e):
            return "{}\n\nQuestion: {}\nA: Let's think step by step.\n".format(self.prompt, e['question'])

        def extract(e, reply):
            e['pred_answer'] = reply
            e['pred_answer_cleaned'] = answer_cleansing(pred=reply, ds_name=self.ds_name)

        todo_list = []
        for i, example in tqdm(enumerate(self.examples), total=len(self.examples)):
            if i % 10 == 0:
                print(f"processing: {i}/{len(self.examples)}")

            if "pred_answer" in example and len(example['pred_answer']) > 10: # contain answer
                continue

            todo_list.append(example)

            if len(todo_list) >= self.args.batch_size or i >= (len(self.examples) - 1):
                if len(todo_list) > 0:
                    # batch_get_api_merge(examples=todo_list, eng=self.args.eng, pre_fun=wrap, post_fun=extract,
                    #                     temperature=self.temperature, timeout=self.args.time_out, max_try=0)
                    gsm8k_test(args, examples = todo_list, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1)
                    todo_list = []

                self.save_data()

                num_correct, num_examples, acc = self.evaluate(i + 1)
                print(
                    "=" * 20 + f"processed: {i}/{len(self.examples)}, acc: {num_correct}/{num_examples}={100 * acc:.2f}")

        self.save_ans_stat()


class SCComplexCoT(ForwardReasoning):
    def __init__(self, args):
        super(SCComplexCoT, self).__init__(args=args)

    def get_method_name(self):
        return "SCComplexCoT"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--eng', default="qwen", type=str)
    parser.add_argument('--model_name', default="Qwen/Qwen2.5-0.5B-Instruct", type=str)
    parser.add_argument('--ds', default="MATH", type=str)
    parser.add_argument('--iter', default="0", type=str)
    parser.add_argument('--temp', default=0.7, help="temperature", type=float)
    parser.add_argument('--method_name', default="SCComplexCoT", type=str)
    parser.add_argument('--cont', action='store_true', help="true=continue previous fetching, default=false")
    parser.add_argument('--num_repeat', default=3, type=int, help="for self-consistency")
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--time_out', default=30, type=int)

    args = parser.parse_args()


    method = SCComplexCoT(args)

    method.fetch_data_from_openai()
    print("final evaluation")
    num_correct, num_question, acc = method.evaluate(len(method.examples))
    msg = f"finished acc: {100 * num_correct / num_question:.4f}"
    get_right_data(args)