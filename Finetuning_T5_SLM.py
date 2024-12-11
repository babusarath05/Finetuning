import os
os.environ['HF_HOME'] = '/Downloads/SB/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import login
login(token="")

#------------------Fine Tuning--------------#
import nltk
import evaluate
import numpy as np
from datasets import load_dataset,Dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME,legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME,device_map="auto",
torch_dtype="auto")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# # Acquire the training data from Hugging Face
# DATA_NAME = "MuskumPillerum/General-Knowledge"
# dataset = load_dataset(DATA_NAME,split='train')

dataset = load_dataset("b-mc2/sql-create-context", split="train")
df = dataset.to_pandas()
df['question']=[f"context:{i}\n\nquestion:{j}" for i,j in df[['context','question']].values]
del df['context']
# df = dataset.to_pandas()
# df = df.dropna()

dataset = Dataset.from_pandas(df)
# dataset = dataset.train_test_split(test_size=0.3)
print(dataset)

# We prefix our tasks with "answer the question"
# prefix = "Please answer this question: "
prefix = ""

# Define the preprocessing function
def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = [prefix + doc for doc in examples["question"]]
   model_inputs = tokenizer(inputs, max_length=512, truncation=True)
  
   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target=examples["answer"], 
                      max_length=512,         
                      truncation=True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

# Map the preprocessing function across our dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

nltk.download("punkt",quiet=True)
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  
   return result

# Global Parameters
L_RATE = 3e-3
BATCH_SIZE = 10
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 3

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="SQLT5",
   # evaluation_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   #per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False ,generation_max_length=512
)

trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset,#["train"],
   # eval_dataset=tokenized_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)

trainer.train()

# Save model
trainer.save_model()

# push to hub
login(token="")
trainer.push_to_hub()

#---------------Inference----------#
last_checkpoint = "SQLT5"

finetuned_model = T5ForConditionalGeneration.from_pretrained(last_checkpoint)
tokenizer = T5Tokenizer.from_pretrained(last_checkpoint)

my_question = "What is the largest continent?"
inputs = "Please answer to this question: " + my_question
inputs = "context: CREATE TABLE employee (emp_id VARCHAR,emp_name VARCHAR,emp_salary INTEGER,emp_department VARCHAR);\n\nquestion:How many rows are there?"

inputs = tokenizer(inputs, return_tensors="pt")
outputs = finetuned_model.generate(**inputs,max_new_tokens=150)
answer = tokenizer.decode(outputs[0],skip_special_tokens=True)
print(answer)