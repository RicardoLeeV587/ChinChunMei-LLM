import argparse
import json, os
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--data_file',default=None, type=str,help="Data File that contains insturctions")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu',action='store_true',help='only use CPU for inference')
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import  PeftModel
from tqdm import tqdm, trange

# generation_config = dict(
#     temperature=0.2,
#     top_k=40,
#     top_p=0.9,
#     do_sample=True,
#     num_beams=1,
#     repetition_penalty=1.3,
#     max_new_tokens=400
#     )

 # The prompt template below is taken from llama.cpp
 # and is slightly different from the one used in training.
 # But we find it gives better results
prompt_input = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{output}"
)

sample_data = ["为什么要减少污染，保护环境？"]

def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})

def generate_sample(example, eos_token):
    if example.get("input", "") != "":
        example["instruction"] = example["instruction"] + "\n" + example["input"]
    return prompt_input.format_map(example) + eos_token

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)
    eos_token = tokenizer.eos_token
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=False,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size!=tokenzier_vocab_size:
        assert tokenzier_vocab_size > model_vocab_size
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model,torch_dtype=load_type,device_map='auto',)
    else:
        model = base_model

    if device==torch.device('cpu'):
        model.float()
    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file,'r') as f:
            if args.data_file.endswith("json"):
                content = f.read()
                examples = json.loads(content)
            else:
                examples = [json.loads(l) for l in f.readlines()]
        
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    
    model.eval()
    
    with torch.no_grad(), torch.autocast("cuda"):
        print("Start inference.")
        results = []
        for example in tqdm(examples, desc='Processing'):
            input_text = generate_sample(example, eos_token)
            #print(input_text)
            inputs = tokenizer(input_text,return_tensors="pt")
            #print(inputs["input_ids"])
            outputs = model(inputs["input_ids"], labels=inputs["input_ids"])
            ppl = torch.exp(outputs.loss)
            dic = {}
            dic.update(example)
            dic["ppl"] = float(torch.exp(outputs.loss).cpu())
            results.append(dic)

        with open(args.predictions_file, "w") as writer:
            writer.write(json.dumps(results, ensure_ascii=False, indent=4))

