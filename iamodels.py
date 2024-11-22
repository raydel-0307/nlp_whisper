import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import pickle
import os
import shutil
import json
import whisper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_dirs(ruta,paths=[".cache","fine_tuned_model","dataset","generate_texts","model"]):
    for i in paths:
        try:os.makedirs(f"{ruta}/{i}")
        except Exception as ex:
            if "File exists" in str(ex):pass

def train_model(ruta,model_name):
    create_dirs(ruta)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    with open(f"{ruta}/model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{ruta}/model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Modelo Exportado '{ruta}/model'")

def tokenize_function(examples,tokenizer,):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def create_labels(examples):
    examples['labels'] = examples['input_ids']
    return examples

def fine_tune(ruta,type_,file_name):
    
    with open(f"{ruta}/model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{ruta}/model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    dataset = load_dataset(type_, data_files=f"{ruta}/dataset/{file_name}")
    tokenized_datasets = dataset.map(tokenize_function,fn_kwargs={'tokenizer': tokenizer}, batched=True).map(create_labels)

    training_args = TrainingArguments(
        output_dir=f'{ruta}/.cache/model',
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['train'],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Eval Loss: {eval_results['eval_loss']}")

    trainer.save_model()

    with open(f"{ruta}/fine_tuned_model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    for i in os.listdir(f"{ruta}/.cache/model"):
        os.rename(f"{ruta}/.cache/model/{i}",f"{ruta}/fine_tuned_model/{i}")
    
    shutil.rmtree(f"{ruta}/.cache")

    print(f"FineTuned Completado, Modelo Exportado '{ruta}/fine_tuned_model'")

def generate(ruta, prompt,prompt_type,whisper_model,is_screen=False,max_length=50,temperature=0.7,top_k=1000,top_p=0.9):
    
    output_file = f"{ruta}/generate_texts/response.json"
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as file:
            response = json.load(file)
    else:
        response = []

    model = GPT2LMHeadModel.from_pretrained(f'{ruta}/fine_tuned_model').to(device)

    with open(f"{ruta}/fine_tuned_model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    if prompt_type=="audio":
        wmodel = whisper.load_model(whisper_model)
        prompt = wmodel.transcribe(prompt)["text"]
        
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1
    )

    resp = tokenizer.decode(output[0], skip_special_tokens=True)
    if is_screen:
        print("User:",prompt)
        print("Response:",resp)

    response.append({
        "user":prompt,
        "response":resp
    })

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(response, f, indent=4, ensure_ascii=False)
    return response

