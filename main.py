from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from huggingface_hub import create_repo
import evaluate
from peft import LoraConfig, get_peft_model
from datasets import DatasetDict
import numpy as np
from init_parameters import init_parameters
from data import split_data, set_seed

def train(index,dataset,args):
    model_name_or_path = args.model
    task = args.task
    metric = evaluate.load("accuracy")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        model_inputs = tokenizer(examples['source'], truncation=True, max_length=None)
        if args.dataset == 'glue':
            model_inputs['labels'] = tokenizer(examples['target'], truncation=True, max_length=None)["input_ids"]
        else:
            model_inputs['labels'] = tokenizer([i[0] for i in examples['target']], truncation=True, max_length=None)["input_ids"]
        return model_inputs
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
    config = LoraConfig(
        r=16,
        task_type="SEQ_2_SEQ_LM",
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

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.argmax(preds, axis=-1)
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

        return metric.compute(predictions=decoded_preds, references=decoded_labels)

    try:
        create_repo(repo_id=f"{model_name}-finetuned-lora-{task}-{index}",token="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")
    except:
        pass

    training_args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-lora-{task}-{index}",
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_total_limit=3,
        num_train_epochs=args.epochs,
        remove_unused_columns=False,
        fp16=True,
        push_to_hub=True
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
    trainer.train()
    trainer.push_to_hub()
    return

def main(args):
    (train_ds, test_ds, valid_ds) = split_data(args)
    for i in range(args.num_clients):
        if args.dataset == 'glue':
            dataset = DatasetDict({'train':train_ds[i],'test':test_ds[i],'valid':valid_ds[i]})
        else:
            dataset = DatasetDict({'train':train_ds[i],'valid':valid_ds[i]})
        train(i,dataset,args)
    return

if __name__ == '__main__':
    args = init_parameters()
    set_seed(args.seed)
    main(args)
