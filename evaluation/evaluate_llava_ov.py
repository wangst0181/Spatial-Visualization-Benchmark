import os
import json
import re
from PIL import Image
from argparse import ArgumentParser
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from tqdm import tqdm
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

def dump_jsonl(save_path, data):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def get_response(model, processor, image_path, text):
    conversation = [
            {
                "role": "user",
                "content": [
                        {"type": "text", "text": text},
                        {"type": "image"},
                    ],
            }
        ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    raw_image = Image.open(image_path)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=4096, do_sample=False) 
    decoded_output = processor.decode(output[0], skip_special_tokens=True)
    if "assistant" in decoded_output:
        reply_start = decoded_output.index("assistant") + len("assistant")
        model_reply = decoded_output[reply_start:].strip()
    else:
        model_reply = decoded_output.strip()
    return model_reply

def get_data_id(item):
    category = item["Category"]
    task = item["Task"]
    level = item["Level"]
    image_id = item['Image_id']
    return f"{category}-{task}-{level}-{image_id}" 

def evaluate(args, model_path, model, processor, data):
    ops = ["A", "B", "C", "D"]
    think_pattern = r'<think>(.*?)</think>'  
    answer_pattern = r'<answer>(.*?)</answer>' 
    prompt = "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
    
    save_path = f"{args.results_dir}/results_{model_path.split(os.sep)[-1]}.jsonl"
    print(save_path)
    result_file = open(save_path, "w", encoding='utf-8')
    
    for idx, item in enumerate(data):
        print(f"{idx}/{len(data)}")
        
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
            response = get_response(model, processor, image_path, text)
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


def modify(args):
    model_paths = args.model_paths
    
    SpatialViz_Bench = load_dataset(args.benchmark_test_path)
    data = SpatialViz_Bench["test"]
    
    for model_path in model_paths:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto'
        ).eval()

        processor = AutoProcessor.from_pretrained(model_path)
        
        ops = ["A", "B", "C", "D"]
        think_pattern = r'<think>(.*?)</think>'  
        answer_pattern = r'<answer>(.*?)</answer>' 
        prompt = "You should first provide a reasoning process, then provide a single option(A, B, C or D) as the final answer. The reasoning process and the answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process</think>, <answer>answer</answer>.\n"
        
        save_name = model_path.split('/')[-1]
        result_file_path = f"{args.results_dir}/results_{save_name}.jsonl"
        if os.path.exists(result_file_path):
            print("loading existed result file......")
            results_data = load_jsonl(result_file_path)
        else:
            print("no existed result file, directely evaluating......")
            evaluate(args, model_path, model, processor, data)
            return True
        
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
                        print("modify failed response......", data_id)
                        result = item.copy()
                    elif task == "CubeUnfolding" and level in ["Level1", "Level2"]:
                        print("modify error promblem of CubeUnfolding......", data_id)
                        result = item.copy()
                    else:
                        print("copying successful response to new file......", data_id)
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
                response = get_response(model, processor, image_path, text)
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
        
        del model
        del processor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+', required=True,
                        default=["SpatialViz-Bench/models/llava-onevision/llava-onevision-qwen2-7b-ov-hf"],
                        help="List of model names or local paths to the models")
    parser.add_argument('--benchmark_test_path', type=str, required=False, 
                        default="SpatialViz-Bench/SpatialViz_Bench_images")
    parser.add_argument('--results_dir', type=str, required=False,
                        default="SpatialViz-Bench/results")
    args = parser.parse_args()
    print("Parsed args:", args)
    modify(args)
