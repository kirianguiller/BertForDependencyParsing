import os
from typing import Dict, List

from transformers.data import data_collator
# os.environ['WANDB_MODE'] = 'dryrun' # deactivate wandb syncing while preserving offline saving
# os.environ['WANDB_DISABLED'] = "True" # deactivating all wandb activities with huggingface
# print("KK GETENV", os.getenv('WANDB_MODE'))
from src.BertForSyntacticParsing import BertForSyntacticParsing, BertConfig, ConlluDataset, Args
from transformers import Trainer, TrainingArguments, AutoTokenizer

import json
import torch

with open("annotation_schema.json") as f:
    schema = json.load(f)

# num_labels = 5
# num_uposs = 10
# num_deprels = 15
# model_config = BertConfig(num_labels=5, num_uposs = 10)
model = BertForSyntacticParsing.from_pretrained(
    "bert-base-cased", num_uposs=len(schema["upos"]), num_deprels=len(schema["deprel"]))

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


args = Args(list_deprel_main=schema["deprel"], list_pos=schema["upos"])


path_conllu_train = "/scratch/kgerdes/kirian/experiences_mémoire/7_unsupervised_FT/train/pcm_nsc-sud-train.conllu"
path_conllu_eval = "/scratch/kgerdes/kirian/experiences_mémoire/7_unsupervised_FT/train/pcm_nsc-sud-dev.conllu"

dataset_train = ConlluDataset(path_conllu_train, tokenizer, args)
dataset_eval = ConlluDataset(path_conllu_eval, tokenizer, args)

# print("KK dataset_eval", dataset_eval, dir(dataset_eval))

training_args = TrainingArguments(
    output_dir="./training_args/",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    eval_steps=100,
    # logging_first_step=True,
    learning_rate=7e-5,
    evaluation_strategy="steps"
)

def dummy_data_collector(features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    batch : Dict[str, torch.Tensor] = {}
    batch['input_ids'] = torch.stack([f[0] for f in features])
    batch['attention_mask'] = torch.stack([f[2] for f in features])
    
    subwords_start = torch.stack([f[1] for f in features])
    idx_convertor = torch.stack([f[3] for f in features])
    poss = torch.stack([f[4] for f in features])
    heads = torch.stack([f[5] for f in features])
    deprels_main = torch.stack([f[6] for f in features])
    # print("KK subwords_start", subwords_start.size())
    # print("KK idx_convertor", idx_convertor.size())
    batch['labels'] = torch.stack([subwords_start, idx_convertor, poss, heads, deprels_main], dim=1)
    
    # print("KK labels.size(", batch["labels"].size())
    # input("input")
    return batch

# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    data_collator=dummy_data_collector,
    # optimizers=(optimizer, None)
)

trainer.train()

# # print(dir(model))
model.save_pretrained("./save_test/")

# # print(dir(model))
# # print("model labels", model.config.num_labels)
# # print("model num_uposs", model.config.num_uposs)
