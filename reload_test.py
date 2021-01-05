from src.BertForSyntacticParsing import BertForTokenClassification


model = BertForTokenClassification.from_pretrained('./save_test_2/')

print(dir(model))
print("model labels", model.config.num_labels)
print("model num_uposs", model.config.num_uposs)