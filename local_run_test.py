import os
from typing import Dict, List

from transformers.data import data_collator
os.environ['WANDB_MODE'] = 'dryrun' # deactivate wandb syncing while preserving offline saving
os.environ['WANDB_DISABLED'] = "True" # deactivating all wandb activities with huggingface
from src.BertForSyntacticParsing import BertForSyntacticParsing, BertConfig, ConlluDataset, Args, dep_parse_data_collator
from transformers import Trainer, TrainingArguments, AutoTokenizer

import json


BERT_MODEL = "camembert-base"
PATH_ROOT_FOLDER = "/home/wran/Research/memoire/BERT_dep_parsing/BertForSyntacticParsing/mock_data"

path_annotation_schema = os.path.join(PATH_ROOT_FOLDER, "annotation_schema.json")
with open(path_annotation_schema) as f:
    schema = json.load(f)

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



# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    data_collator=dep_parse_data_collator,
    # optimizers=(optimizer, None)
)

trainer.train()

model.save_pretrained("./save_test/")

