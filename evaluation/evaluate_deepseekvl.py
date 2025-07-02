# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os

from argparse import ArgumentParser
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM
import PIL.Image
from datasets import load_dataset

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor

import json, re
from tqdm import tqdm

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

def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image>\nExtract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """

    pil_images = []

    for message in conversations:
        if "images" not in message:
            continue

        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)

    return pil_images

def make_conversation(image_path, text):
    conversation = [
        {
            "role": "<|User|>",
            "content": f"{text}\n<image>\n",
            "images": [
                image_path
            ],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    return conversation

def get_response(image_path, text, vl_chat_processor, vl_gpt, tokenizer, dtype):
    conversation = make_conversation(image_path, text)
    pil_images = load_pil_images(conversation)
    print(f"len(pil_images) = {len(pil_images)}")
    prepare_inputs = vl_chat_processor.__call__(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(vl_gpt.device, dtype=dtype)
    
    with torch.no_grad():
        if args.chunk_size == -1:
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            past_key_values = None
        else:
            # incremental_prefilling when using 40G GPU for vl2-small
            inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                chunk_size=args.chunk_size
            )

        # run the model to get the response
        outputs = vl_gpt.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,

            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=4096,

            # do_sample=False,
            # repetition_penalty=1.1,

            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,

            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
    return answer

def get_data_id(item):
    category = item["Category"]
    task = item["Task"]
    level = item["Level"]
    image_id = item['Image_id']
    return f"{category}-{task}-{level}-{image_id}" 

def evaluate(args, model_path, vl_chat_processor, vl_gpt, tokenizer, dtype, data):
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
            response = get_response(image_path, text, vl_chat_processor, vl_gpt, tokenizer, dtype)
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
    dtype = torch.bfloat16
    SpatialViz_Bench = load_dataset(args.benchmark_test_path)
    data = SpatialViz_Bench["test"]

    for model_path in args.model_paths:
        # specify the path to the model
        vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype
        )
        vl_gpt = vl_gpt.cuda().eval()
        
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
            evaluate(args, model_path, vl_chat_processor, vl_gpt, tokenizer, dtype, data)
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
                    # elif "Response" in result and "<answer>" not in result["Response"]:
                    #     print("modify short-cut response......", data_id)
                    #     result = item.copy()
                    elif task == "CubeUnfolding" and level in ["Level1", "Level2"]:
                        # print("modify error promblem of CubeUnfolding......", data_id)
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
                response = get_response(image_path, text, vl_chat_processor, vl_gpt, tokenizer, dtype)
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
        
        del vl_gpt
        del tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+', required=False,
                        default=["SpatialViz-Bench/models/deepseek/deepseek-vl2-small"],
                        help="model name or local path to the model")
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="chunk size for the model for prefiiling. "
                             "When using 40G gpu for vl2-small, set a chunk_size for incremental_prefilling."
                             "Otherwise, default value is -1, which means we do not use incremental_prefilling.")
    parser.add_argument('--benchmark_test_path', type=str, required=False, 
                        default="SpatialViz-Bench/SpatialViz_Bench_images")
    parser.add_argument('--results_dir', type=str, required=False,
                        default="SpatialViz-Bench/results")
    args = parser.parse_args()
    print(args)
    modify(args)
