from .file_utils import is_transformers_available

from .modeling_bert import BertForSyntacticParsing
from .configuration_bert import BertConfig
from .load_data_utils import ConlluDataset, Args

if not is_transformers_available():
    raise "The `transformers` package was not found"
