import os
os.environ['HF_HOME'] = '/Downloads/SB/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import login
login(token="")



#---------------Inference----------------------#
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id

input_text = "Write a poem about Love"
# Prepare the input text you want to generate predictions for
inputs = tokenizer(input_text, return_tensors='pt').to("cuda")

# Generate Text
outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

# ------------Fine Tuning-------------#
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset,Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

# Set the EOS token as the padding token
tokenizer.pad_token = tokenizer.eos_token

# Load Dataset
# dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")
dataset = load_dataset("b-mc2/sql-create-context", split="train")
df = dataset.to_pandas()
df['messages']=[f"context:{i}\n\nquestion:{j}\n\nanswer:{k}" for i,j,k in df[['context','question','answer']].values]
df = df.drop(columns=['context','question','answer'],axis=1)
dataset = Dataset.from_pandas(df)
print(dataset[3]['messages'])
dataset = dataset.train_test_split(test_size=0.2)


# Tokenize the dataset
def tokenize_function(examples):
    inputs =  tokenizer(examples['messages'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='SQLgpt2',
    eval_strategy='epoch',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs'
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

trainer.train()

# push to hub
login(token="")
trainer.push_to_hub()

# trainer.save_model()

model.save_pretrained('SQLgpt2')
tokenizer.save_pretrained('SQLgpt2')

#---------------Inference----------------------#
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = 'SQLgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id

input_text = "context: CREATE TABLE employee (emp_id VARCHAR,emp_name VARCHAR,emp_salary INTEGER,emp_department VARCHAR);\n\nquestion:What is the highest salary?"
# Prepare the input text you want to generate predictions for
inputs = tokenizer(input_text, return_tensors='pt').to("cuda")

# Generate Text
outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)