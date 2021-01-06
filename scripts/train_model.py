import sys
PATH_PACKAGE = "/home/wran/Research/memoire/BERT_dep_parsing/BertForSyntacticParsing/src/"
sys.path.insert(0, PATH_PACKAGE)


import os
from BertForSyntacticParsing import BertForSyntacticParsing, ConlluDataset, Args, dep_parse_data_collator, ModelFolderHandler, dep_parse_metrics_computor
import json
from transformers import Trainer, TrainingArguments, AutoTokenizer

# # deactivate wandb syncing while preserving offline saving
# os.environ['WANDB_MODE'] = 'dryrun'
# # deactivating all wandb activities with huggingface
os.environ['WANDB_DISABLED'] = "False"



BERT_MODEL = "camembert-base"
PATH_ROOT_FOLDER = "/home/wran/Research/memoire/BERT_dep_parsing/BertForSyntacticParsing/mock_data"
SEED = 42


MODEL_NAME = f"pretrain__{BERT_MODEL}__seed_{SEED}"

path_handler = ModelFolderHandler(PATH_ROOT_FOLDER, MODEL_NAME)


with open(path_handler.annotation_schema) as f:
    schema = json.load(f)

model = BertForSyntacticParsing.from_pretrained(
    BERT_MODEL, num_uposs=len(schema["upos"]), num_deprels=len(schema["deprel"]))

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)


args = Args(maxlen=model.maxlen, list_deprel_main=schema["deprel"], list_pos=schema["upos"])


train_dataset = ConlluDataset(path_handler.train_folder, tokenizer, args)
eval_dataset = ConlluDataset(path_handler.eval_folder, tokenizer, args)


training_args = TrainingArguments(
    output_dir=path_handler.training_args,
    overwrite_output_dir=True,
    num_train_epochs=300,
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    save_steps=300,
    save_total_limit=2,
    logging_steps=30,
    eval_steps=30,
    # logging_first_step=True,
    learning_rate=1e-4,
    # evaluation_strategy="steps",
    seed=SEED,
)


# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=dep_parse_data_collator,
    compute_metrics=dep_parse_metrics_computor,
    # optimizers=(optimizer, None)
)

trainer.train()

model.save_pretrained(path_handler.saved_model)
