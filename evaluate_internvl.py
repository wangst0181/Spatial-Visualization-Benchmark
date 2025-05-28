from argparse import ArgumentParser
import os
import re
import json

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset

from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_response(model, tokenizer, generation_config, image_path, text):
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    response = model.chat(tokenizer, pixel_values, text, generation_config)
    return response

def get_data_id(item):
    category = item["Category"]
    task = item["Task"]
    level = item["Level"]
    image_id = item['Image_id']
    return f"{category}-{task}-{level}-{image_id}"    

def evaluate(args, model_path, model, tokenizer, generation_config, data):
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
            response = get_response(model, tokenizer, generation_config, image_path, text)
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
        
        # print(response)
        # print(think_match)
        # print(answer_match)
        
        result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        result_file.flush()

def modify(args):
    model_paths = args.model_paths
    SpatialViz_Bench = load_dataset(args.benchmark_test_path)
    data = SpatialViz_Bench["test"]
    
    for model_path in model_paths:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map='auto').eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=4096, do_sample=True)
        
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
            evaluate(args, model_path, model, tokenizer, generation_config, data)
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
                        # print("modify failed response......", data_id)
                        result = item.copy()
                    elif task == "CubeUnfolding" and level in ["Level1", "Level2"]:
                        # print("modify error promblem of CubeUnfolding......", data_id)
                        result = item.copy()
                    elif "Response" in result and "<answer>" not in result["Response"]:
                        # print("modify short-cut response......", data_id)
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
                response = get_response(model, tokenizer, generation_config, image_path, text)
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
        del tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+', required=False,
                        default=["SpatialViz-Bench/models/internvl/internvl2_5-8b"],
                        help="List of model names or local paths to the models")
    parser.add_argument('--benchmark_test_path', type=str, required=False, 
                        default="SpatialViz-Bench/SpatialViz_Bench_images")
    parser.add_argument('--results_dir', type=str, required=False,
                        default="SpatialViz-Bench/results")
    args = parser.parse_args()
    print("Parsed args:", args)
    modify(args)