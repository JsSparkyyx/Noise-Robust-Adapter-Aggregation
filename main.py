from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import load_dataset
import evaluate
import torch
from peft import LoraConfig, get_peft_model
import numpy as np
from init_parameters import init_parameters
import random

def split_data(args):
    data = []
    task = args.task

    if args.dataset == 'glue':
        dataset = load_dataset("JsSparkYyx/processed_glue", task)
    return data

def main(args):
    model_name_or_path = args.model
    task = args.task

    if args.dataset == 'glue':
        dataset = load_dataset("JsSparkYyx/processed_glue", task)
    metric = evaluate.load("sacrebleu")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q","k","v","o"],
        lora_dropout=0.1,
        bias="none",
    )

    model_name = model_name_or_path.split("/")[-1]
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
    model = get_peft_model(model, config)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    model.print_trainable_parameters()

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        model_inputs = tokenizer(examples['source'], truncation=True, max_length=None)
        model_inputs['labels'] = tokenizer(examples['target'], truncation=True, max_length=None)["input_ids"]
        return model_inputs

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100s in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    training_args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-lora-{task}",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(tokenized_datasets["test"])
    trainer.evaluate()
    return

if __name__ == '__main__':
    args = init_parameters()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    main(args)
