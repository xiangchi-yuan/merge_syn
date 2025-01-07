import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
import os
from utils.answer_clean_utils import answer_cleansing

MAX_INT = sys.maxsize

ds_path_dict = {
    "GSM8K": "GSM8K/gsm8k_train",
    "MATH": "MATH/MATH_train",
    "GSM8K_rephrased": "GSM8K/gsm8k_train-cleaned_rephrased_questions",
    "MATH_rephrased": "MATH/MATH_train-cleaned_rephrased_questions",
}

def get_right_data(args):
    pre_file = args.save_file
    out_file = args.save_right_file
    with open(pre_file, 'r') as file:
        data = json.load(file)


    # Function to filter and rename keys
    def process_entries(entries):
        processed_entries = []
        right_count = 0
        for entry in entries:
            if entry['pred_answer_cleaned'] == entry['answer']:
                # Renaming 'answer' to 'query' and 'pred_answer' to 'response'
                entry['query'] = entry.pop('question')
                entry['response'] = entry.pop('pred_answer')
                processed_entries.append(entry)
                right_count = right_count + 1
        print("acc:", right_count/len(entries))
        return processed_entries

    # Filter and rename the data
    filtered_renamed_data = process_entries(data)

    # Save the processed data back to a new JSON file
    with open(out_file, 'w') as file:
        json.dump(filtered_renamed_data, file, indent=4)

    print("Processed data has been saved.")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def extract_answer_number_v2(pred, split_str="The answer is"):
    preds = pred.split(split_str)

    pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    return pred

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n

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

def gsm8k_test(args, model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    prompt = get_prompt()
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request like given examples below:\n\n"
        "{}\n### Instruction:\n{{instruction}}\n\n### Response: Let's think step by step.".format(prompt)
    )
    print('promt =====', problem_prompt)

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 循环遍历每个元素
    for item in data:
        temp_instr = problem_prompt.format(instruction=item["question"])
        gsm8k_ins.append(temp_instr)



    gsm8k_ins = gsm8k_ins[start:end]

    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    result = []
    res_completions = []
    for idx, prompt in enumerate(batch_gsm8k_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    # print(len(res_completions))
    # with open('GSM8K/test.json', 'w', encoding='utf-8') as f:
    #     json.dump(res_completions, f, ensure_ascii=False, indent=4)

    for i, item in enumerate(data):
        item['pred_answer'] = res_completions[i]
        item['pred_answer_cleaned'] = answer_cleansing(pred=res_completions[i], ds_name='GSM8K')

    with open(args.save_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-0.5B-Instruct')  # model path
    parser.add_argument("--data_file", type=str, default='GSM8K/gsm8k_train.json')  # data path
    parser.add_argument("--save_file", type=str, default='GSM8K/test.json')  # data path
    parser.add_argument("--save_right_file", type=str, default='GSM8K/right_test.json')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=3000)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(args=args, model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
    get_right_data(args=args)