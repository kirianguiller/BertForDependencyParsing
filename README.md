# BertForDependencyParsing

A transformers extension for training, fine-tuning and using transformers models (BERT-like architecture) for dependency parsing. 

## Installation
To install it on a regular ubuntu system, follow these instructions

### GIT setup
Clone this repository somewhere on your machine
```
git clone https://github.com/kirianguiller/BertForDependencyParsing
```

### python setup
create the virtual environment at the root of this repo
```
python -m venv venv
```

activate the environment
```
source venv/bin/activate
```

install the required pip packages
```
pip install requirements.txt
```

### wandb setup
activate wandb online syncronization
```
wandb on
```

## Model training pipeline 
### Setting the data
Create a data folder where the train/valid/test conllu files, the saved model, the annotation schema and other files will be stored.
For training a model, the minimum required structure is the following :
```
folder/
  train/
    train1.conllu
    train2.conllu
    ...
  eval/
    eval1.conllu
    eval2.conllu
    ...
  test/
    test1.conllu
    test2.conllu
    ...
  
```
### Annotation schema
Compute the annotation schema from the conllu that will be using for training (and fine-tuning) the model. When first training the model, it is important to have an annotation schema that already include the labels (deprel, upos) of the conllu that will be use for fine-tuning as the number of labels in the annotation schema will be corresponding with one of the dimension of the model.

```
TODO : create the script
```

### Train the model
Change the variable in scripts/train_model.py and run the script for training the model

```
python scripts/train_model.py
```

The model will be save in the data folder your provided, in the subfolder models/$model_name
