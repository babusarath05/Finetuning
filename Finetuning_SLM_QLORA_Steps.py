#-----------Install Libraries----------------#
# pip install torch
# pip install transformers
# pip install bitsandbytes
# pip install accelerate
# pip install datasets
# pip install peft
# pip install trl

#------------Set Environment-----------------#
import os
os.environ['HF_HOME'] = '/Downloads/SB/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#-----------Read Token-----------------------#
from huggingface_hub import login
login(token="")

#-------------Model Testing------------------#
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
)

input_text = "Write me a poem about Machine Learning in less than 100 words."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=512)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))

#------------Fine Tuning---------------------#

from datasets import load_dataset,Dataset

# # # Load Dolly Dataset
# dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
dataset = load_dataset("b-mc2/sql-create-context", split="train")
df = dataset.to_pandas()
# df['messages']=[[{"content":i,"role":"user"},{"content":j,"role":"assistant"}] for i,j in df[['prompt','completion']].values]
df['messages']=[[{"content":f"context:{i}\n\nquestion:{j}","role":"user"},{"content":k,"role":"assistant"}] for i,j,k in df[['context','question','answer']].values]
# df = df.drop(columns=['prompt','completion','input_ids','attention_mask','labels'],axis=1)
df = df.drop(columns=['context','question','answer'],axis=1)
dataset = Dataset.from_pandas(df)
print(dataset[3]['messages'])


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Hugging Face model id
model_id = "google/gemma-2-2b-it"
tokenizer_id = "philschmid/gemma-tokenizer-chatml"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={'':torch.cuda.current_device()},
    #attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.padding_side = 'right' # to prevent warnings

from peft import LoraConfig

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=6,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="SQLgemma-2-2b-it", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    #report_to="tensorboard",                # report metrics to tensorboard
)

from trl import SFTTrainer

max_seq_length = 1512 # max sequence length for model and packing of the dataset

# model.resize_token_embeddings(len(tokenizer))

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": False, # No need to add additional separator token
    }
)

# Start training
trainer.train()

# Save model
trainer.save_model()

#-------------Inference------------------#

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,pipeline
from peft import AutoPeftModelForCausalLM
import torch

peft_model_id =  "/Downloads/SB/SQLgemma-2-2b-it"
# Load adapted model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Get token id for end of conversation
# eos_token = tokenizer("",add_special_tokens=False)["input_ids"][0]

prompts = [
    # "What is Fine Tuning? Explain why thats the case and if it was different in the past?",
    # "Write a Python function to calculate the factorial of a number.",
    # "What is the sql query to find the employee with the lowest salary in a table named employee",
    "context:CREATE TABLE city (City_ID VARCHAR, Population INTEGER); CREATE TABLE farm_competition (Theme VARCHAR, Host_city_ID VARCHAR)\n\nquestion:Please show the themes of competitions with host cities having populations larger than 5000."
]

def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.01, top_k=50, top_p=0.95)
    return outputs[0]['generated_text'][len(prompt):].strip()

# Test inference
for prompt in prompts:
    print(f"    Prompt:\n{prompt}")
    print(f"    Response:\n{test_inference(prompt)}")
    print("-"*50)

#------------Merging the Base model with Adapter-----#
base_model_url = "google/gemma-2-2b-it"
new_model_url = "/Downloads/SB/SQLgemma-2-2b-it"
tokenizer_url = "/Downloads/SB/SQLgemma-2-2b-it"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format


# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_url)

base_model_reload= AutoModelForCausalLM.from_pretrained(
    base_model_url,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cpu",
)

model = PeftModel.from_pretrained(base_model_reload, new_model_url)

model = model.merge_and_unload()

model.save_pretrained("SQLGemma-2-2b-it")
tokenizer.save_pretrained("SQLGemma-2-2b-it")

#-----------Final Inference-----------------#
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("SQLGemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "SQLGemma-2-2b-it",
    device_map="cuda",
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompts = [
    # "What is Fine Tuning? Explain why thats the case and if it was different in the past?",
    # "Write a Python function to calculate the factorial of a number.",
    "context: CREATE TABLE employee (emp_id VARCHAR,emp_name VARCHAR,emp_salary INTEGER,emp_department VARCHAR);\n\nquestion:What is the sql query to find the employee name with the lowest salary?",
    # "context:CREATE TABLE city (City_ID VARCHAR, Population INTEGER); CREATE TABLE farm_competition (Theme VARCHAR, Host_city_ID VARCHAR);\n\nquestion:Please show the themes of competitions with host cities having populations larger than 5000."
]

def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.01, top_k=50, top_p=0.95)
    return outputs[0]['generated_text'][len(prompt):].strip()

# Test inference
for prompt in prompts:
    print(f"    Prompt:\n{prompt}")
    print(f"    Response:\n{test_inference(prompt)}")
    print("-"*50)

#-------------Pushing the model to Huggingface Hub-----#
login(token="")
model.push_to_hub("SQLGemma-2-2b-it", use_temp_dir=False)
tokenizer.push_to_hub("SQLGemma-2-2b-it", use_temp_dir=False)

#-----------Testing Inference from HuggingFace hub-----------------#
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
import torch

tokenizer = AutoTokenizer.from_pretrained("Sarathbabu-Karunanithi/SQLGemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "Sarathbabu-Karunanithi/SQLGemma-2-2b-it",
    device_map="cuda",
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompts = [
    # "What is Fine Tuning? Explain why thats the case and if it was different in the past?",
    # "Write a Python function to calculate the factorial of a number.",
    "context: CREATE TABLE employee (emp_id VARCHAR,emp_name VARCHAR,emp_salary INTEGER,emp_department VARCHAR);\n\nquestion:What is the sql query to find the employee name with the lowest salary?",
    # "context:CREATE TABLE city (City_ID VARCHAR, Population INTEGER); CREATE TABLE farm_competition (Theme VARCHAR, Host_city_ID VARCHAR);\n\nquestion:Please show the themes of competitions with host cities having populations larger than 5000."
]

def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.01, top_k=50, top_p=0.95)
    return outputs[0]['generated_text'][len(prompt):].strip()

# Test inference
for prompt in prompts:
    print(f"    Prompt:\n{prompt}")
    print(f"    Response:\n{test_inference(prompt)}")
    print("-"*50)

#--------------------Quantizing and uploading to Huggingface Hub--------------#
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Hugging Face model id
model_id = "Sarathbabu-Karunanithi/SQLGemma-2-2b-it"
# tokenizer_id = "philschmid/gemma-tokenizer-chatml"
tokenizer_id = "Sarathbabu-Karunanithi/SQLGemma-2-2b-it"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={'':torch.cuda.current_device()},
    #attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

model.save_pretrained("SQLGemma-2-2b-it")
tokenizer.save_pretrained("SQLGemma-2-2b-it")
