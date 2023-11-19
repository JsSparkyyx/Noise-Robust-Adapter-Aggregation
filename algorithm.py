from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict

def average_aggreation(lora_adaptors):
    weight = 1/len(lora_adaptors)
    final_state_dict = {}
    keys = lora_adaptors[0].keys()
    for i, lora_adaptor in enumerate(lora_adaptors):
        if i == 0:
            for key in keys:
                final_state_dict[key] = weight * lora_adaptor[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weight * lora_adaptor[key]
                )
    return final_state_dict

def load_lora_adaptors(args):
    lora_adaptors = []
    for i in range(args.num_clients):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, return_dict=True)
        lora_model = PeftModel.from_pretrained(base_model,f'JsSparkYyx/flan-t5-base-finetuned-lora-{args.task}-{i}')
        lora_adaptors.append(get_peft_model_state_dict(lora_model))
    return lora_adaptors

def update_lora_weights(model, state_dict):
    set_peft_model_state_dict(model,state_dict)
    model = model.merge_and_unload()
    return model