import os
from typing import Dict, List

from transformers.data import data_collator
os.environ['WANDB_MODE'] = 'dryrun' # deactivate wandb syncing while preserving offline saving
os.environ['WANDB_DISABLED'] = "True" # deactivating all wandb activities with huggingface
from src.BertForSyntacticParsing import BertForSyntacticParsing, BertConfig
from src.BertForSyntacticParsing import ConlluDataset, Args
from transformers import Trainer, TrainingArguments, AutoTokenizer

import json
import torch

with open("annotation_schema.json") as f:
    schema = json.load(f)

BERT_MODEL = "camembert-base"

# model_config = BertConfig(num_labels=5, num_uposs = 10)
model = BertForSyntacticParsing.from_pretrained(
    BERT_MODEL, num_uposs=len(schema["upos"]), num_deprels=len(schema["deprel"]))

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)


args = Args(list_deprel_main=schema["deprel"], list_pos=schema["upos"])


path_conllu_train = "./data/test.conllu"
path_conllu_eval = "./data/test.conllu"
dataset_train = ConlluDataset(path_conllu_train, tokenizer, args)
dataset_eval = ConlluDataset(path_conllu_train, tokenizer, args)


training_args = TrainingArguments(
    output_dir="./training_args/",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_steps=100,
    save_total_limit=2,
    logging_steps=1,
    eval_steps=1,
    # logging_first_step=True,
    learning_rate=2e-5,
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

model.save_pretrained("./save_test/")

