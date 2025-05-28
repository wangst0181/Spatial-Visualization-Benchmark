import json
import os
import re
import base64
import math
import time
import random
from tqdm import tqdm
from openai import OpenAI
from copy import deepcopy
from argparse import ArgumentParser
from datasets import load_dataset

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            results.append(data)
    return results

def dump_json(save_path, data):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        
def dump_jsonl(save_path, data):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_client(openai_api_key, openai_api_base):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def get_response(key, url, model_name_or_path, image_path, text):
    client = create_client(key, url)
    encoded_image = encode_image(image_path)
    image_type = image_path.split('.')[-1]
    messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_type};base64,{encoded_image}",
                            },
                        },
                        {"type": "text", "text": text},
                    ],
                }]
    if 'omni' not in model_name_or_path:
        chat_response = client.chat.completions.create(
            model=model_name_or_path, 
            messages=messages,
        )

        if chat_response.choices is None:
            return chat_response
        elif hasattr(chat_response.choices[0].message, 'reasoning_content') and chat_response.choices[0].message.reasoning_content is not None:
            return f"<think>{chat_response.choices[0].message.reasoning_content}</think>" + chat_response.choices[0].message.content
        else:
            return chat_response.choices[0].message.content
    else:
        chat_response = client.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True}
        )
        response = ""
        for chunk in chat_response:
            if chunk.choices:
                if hasattr(chunk.choices[0].delta, 'content')  and chunk.choices[0].delta.content is not None:
                    response += chunk.choices[0].delta.content
                if hasattr(chunk.choices[0].delta, 'reasoning_content')  and chunk.choices[0].delta.reasoning_content is not None:
                    response += chunk.choices[0].delta.reasoning_content
        return response

def get_response_qwenvl(model_path, image_path, text):
    return get_response(key="EMPTY", url="http://localhost:8000/v1", model_name_or_path=model_path, image_path=image_path, text=text)

def get_response_internvl(model_path, image_path, text):
    return get_response(key="EMPTY", url="http://0.0.0.0:8000/v1", model_name_or_path=model_path, image_path=image_path, text=text)

def get_response_closed_models(model_name, image_path, text, args):
    if model_name in ["qvq-72b-preview", "qwen2.5-vl-72b-instruct", "qwen2.5-vl-32b-instruct", "qwen2.5-vl-7b-instruct", "qwen2.5-vl-3b-instruct", "qwen-vl-max-latest", "qwen2.5-omni-7b", "qwen-omni-turbo"]:
        return get_response(key=args.qwen_key, 
                            url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
                            model_name_or_path=model_name, 
                            image_path=image_path, text=text)
    elif model_name in ["doubao-1-5-vision-pro-32k-250115"]:
        return get_response(key=args.doubao_key, 
                            url="https://ark.cn-beijing.volces.com/api/v3", 
                            model_name_or_path=model_name, 
                            image_path=image_path, text=text)
    elif model_name in ["gemini-1.5-pro", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-03-25"]:
        return get_response(key=args.gemini_key, 
                            url="https://generativelanguage.googleapis.com/v1beta/openai/", 
                            model_name_or_path=model_name, 
                            image_path=image_path, text=text)
    elif model_name in ["gpt-4o", "o1"]:
        return get_response(key=args.openai_key, 
                            url="https://api.openai.com/v1", 
                            model_name_or_path=model_name, 
                            image_path=image_path, text=text)
    elif model_name in ["moonshotai/moonlight-16b-a3b-instruct:free", "moonshotai/kimi-vl-a3b-thinking:free", "meta-llama/llama-4-maverick", "meta-llama/llama-4-scout", "anthropic/claude-3.5-sonnet", "openai/o1", "openai/gpt-4o"]:
        return get_response(key=args.openrouter_key, 
                            url="https://openrouter.ai/api/v1", 
                            model_name_or_path=model_name, 
                            image_path=image_path, text=text)
    else:
        raise ValueError("You should add your tested models here.")
    
    
def evaluate(model_list, data, args):
    ops = ["A", "B", "C", "D"]
    think_pattern = r'<think>(.*?)</think>'  
    answer_pattern = r'<answer>(.*?)</answer>' 
    prompt = "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
        
    for model in model_list:
        print(model)
        
        save_name = model.split('/')[-1]
        save_path = f"{args.results_dir}/results_{save_name}.jsonl"
        print(save_path)

        result_file = open(save_path, "w", encoding='utf-8')
        
        for item in tqdm(data):
            category = item["Category"]
            task = item["Task"]
            level = item["Level"]
            image_id = item['Image_id']
            image_path = f"{args.benchmark_test_path}/{category}/{task}/{level}/{image_id}.png"
            
            question = item["Question"]
            choices = item["Choices"]
            choice_text = ""
            for i, choice in enumerate(choices):
                choice_text += ops[i] + '. ' + choice + '\n'
            text = prompt + "Qustion: " + question + "\n" + choice_text
            
            result = item.copy()
            
            try:
                response = get_response_closed_models(model, image_path, text, args)
                think_match = re.search(think_pattern, response, re.DOTALL)
                answer_match = re.search(answer_pattern, response, re.DOTALL)
                if think_match and answer_match:
                    result.update({
                        "InputText": text,
                        "ThinkingProcess": think_match.group(1).strip(), 
                        "FinalAnswer": answer_match.group(1).strip() 
                    })
                else:
                    result.update({
                        "InputText": text,
                        "Response": response
                    })
            except Exception as e:
                print(image_path)
                print(e)
                result = item.copy()
                result.update({
                    "InputText": text,
                    "Response": "Failed!!!"
                })
                
            result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            result_file.flush()
            time.sleep(1)

def get_data_id(item):
    category = item["Category"]
    task = item["Task"]
    level = item["Level"]
    image_id = item['Image_id']
    return f"{category}-{task}-{level}-{image_id}"

def modify(args):
    SpatialViz_Bench = load_dataset(args.benchmark_test_path)
    data = SpatialViz_Bench["test"]
    
    for model in args.model_list:
        print(model)
        ops = ["A", "B", "C", "D"]
        think_pattern = r'<think>(.*?)</think>'  
        answer_pattern = r'<answer>(.*?)</answer>' 
        prompt = "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"

        save_name = model.split('/')[-1]
        
        result_file_path = f"{args.results_dir}/results_{save_name}.jsonl"
        if os.path.exists(result_file_path):
            print("loading existed result file......")
            results_data = load_jsonl(result_file_path)
        else:
            print("no existed result file, directely evaluating......")
            evaluate([model], data, args)
            continue
        
        save_path = f"{args.results_dir}/results_{save_name}_modify.jsonl"
        print(save_path)
        new_result_file = open(save_path, "w", encoding='utf-8')
        
        idx = 0
        for item in tqdm(data):
            category = item["Category"]
            task = item["Task"]
            level = item["Level"]
            image_id = item['Image_id']
            image_path = f"{args.benchmark_test_path}/{category}/{task}/{level}/{image_id}.png"
            
            question = item["Question"]
            choices = item["Choices"]
            
            choice_text = ""
            for i, choice in enumerate(choices):
                choice_text += ops[i] + '. ' + choice + '\n'
            text = prompt + "Qustion: " + question + "\n" + choice_text
            
            data_id = get_data_id(item)
            if idx < len(results_data): 
                result = results_data[idx]
                result_id = get_data_id(result)
                if data_id == result_id:
                    idx += 1
                    if "Response" in result and result["Response"] == "Failed!!!":
                        # print("modify failed response......", data_id)
                        result = item.copy()
                    else:
                        # print("copying successful response to new file......", data_id)
                        new_result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        new_result_file.flush()
                        continue 
                else:
                    # print("new evaluation......", data_id)   
                    result = item.copy()
            else: 
                # print("new evaluation......", data_id)   
                result = item.copy()
                
            try:
                response = get_response_closed_models(model, image_path, text, args)
                # print(response)
                think_match = re.search(think_pattern, response, re.DOTALL)
                answer_match = re.search(answer_pattern, response, re.DOTALL)
                
                if think_match and answer_match:
                    result.update({
                        "InputText": text,
                        "ThinkingProcess": think_match.group(1).strip(), 
                        "FinalAnswer": answer_match.group(1).strip() 
                    })        
                else:
                    result.update({
                        "InputText": text,
                        "Response": response
                    })
            except Exception as e:
                print(image_path)
                print(e)
                result = item.copy()
                result.update({
                    "InputText": text,
                    "Response": "Failed!!!"
                })
            new_result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            new_result_file.flush()
            time.sleep(1)
        
def get_answer(file_path, args):
    print(file_path)
    results = load_jsonl(file_path)
    
    positives = []
    negatives = []
    counting = {}
    counting["total"] = {"correct_num": 0, "total_num": 0}
    for i, result in enumerate(results):
        counting["total"]["total_num"] += 1
        
        category = result["Category"]
        task = result["Task"]
        level = result["Level"]
        key = f"{category}-{task}-{level}"
        if category not in counting:
            counting[category] = {"correct_num": 0, "total_num": 1}
        else:
            counting[category]["total_num"] += 1
        
        if task not in counting:
            counting[task] = {"correct_num": 0, "total_num": 1}
        else:
            counting[task]["total_num"] += 1
        
        if key not in counting:
            counting[key] = {"correct_num": 0, "total_num": 1}
        else:
            counting[key]["total_num"] += 1
        
        ground_truth = result['Answer']
        
        op = []
        if "FinalAnswer" in result:
            pred_answer = result['FinalAnswer'].split('.')[0]
            op = re.findall(r'[A-D]', pred_answer)
        elif "Response" in result:
            response = result['Response']
            final_answer_patterns = ["<answer>", "Answer:", "Final answer", "final answer", "Final Answer", "the answer is", "The answer is", "correct answer", "Correct answer", "Correct Answer", "答案" "correct path"]
            if len(response) == 1:
                op = re.findall(r'[A-D]', response)
            else:
                for pattern in final_answer_patterns:
                    if pattern in response:
                        response = response.split(pattern)[-1].strip()
                        op = re.findall(r'[A-D]', response.split('.')[0])
                        break
        
        op = list(set(op))
        if len(op) == 1 and ground_truth == op[0].upper():
            counting["total"]["correct_num"] += 1
            counting[category]["correct_num"] += 1
            counting[task]["correct_num"] += 1
            counting[key]["correct_num"] += 1
            
            instance = {"DataID": f"{key}-{result['Image_id']}", "InputText": result["InputText"], "Answer": result["Answer"]}
            if "Response" in result:
                instance.update({"Response": result["Response"]})
            elif "ThinkingProcess" in result:
                instance.update({"ThinkingProcess": result["ThinkingProcess"]})
                instance.update({"FinalAnswer": result["FinalAnswer"]}) 
            positives.append(instance)
        else:
            instance = {"DataID": f"{key}-{result['Image_id']}", "InputText": result["InputText"], "Answer": result["Answer"]}
            if "Response" in result:
                instance.update({"Response": result["Response"]})
            elif "ThinkingProcess" in result:
                instance.update({"ThinkingProcess": result["ThinkingProcess"]})
                instance.update({"FinalAnswer": result["FinalAnswer"]}) 
            
            negatives.append(instance)
     
    print(len(positives), len(negatives))
    for k in counting:
        counting[k]["acc"] = "{:.2f}".format(counting[k]["correct_num"] / counting[k]["total_num"] * 100)
    
    
    os.makedirs(f"{args.results_dir}/samples", exist_ok=True)
    os.makedirs(f"{args.results_dir}/counting", exist_ok=True)
    
    dump_json(f"{args.results_dir}/samples/{file_path.split(os.sep)[-1][:-6]}_samples.json", {"Positives": positives, "Negatives": negatives})
    dump_json(f"{args.results_dir}/counting/{file_path.split(os.sep)[-1][:-6]}_counting.json", counting)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_list", type=str, nargs='+', required=False,
                        default=["qwen2.5-vl-3b-instruct"],
                        help="List of model names or local paths to the models")
    parser.add_argument('--benchmark_test_path', type=str, required=False, 
                        default="SpatialViz-Bench/SpatialViz_Bench_images")
    parser.add_argument('--results_dir', type=str, required=False,
                        default="SpatialViz-Bench/results")
    parser.add_argument('--qwen_key', type=str, required=False,
                        help="Your api key for Qwen models.")
    parser.add_argument('--doubao_key', type=str, required=False,
                        help="Your api key for Doubao models.")
    parser.add_argument('--openai_key', type=str, required=False,
                        help="Your api key for Openai models.")
    parser.add_argument('--gemini_key', type=str, required=False,
                        help="Your api key for Gemini models.")
    parser.add_argument('--openrouter_key', type=str, required=False,
                        help="Your api key for Openrouter.")
    args = parser.parse_args()
    print("Parsed args:", args)
    modify(args)
    
    
    for result_file_path in os.listdir(args.results_dir):
        get_answer(f"{args.results_dir}/{result_file_path}", args)
    