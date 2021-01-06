from .file_utils import is_transformers_available

from .modeling_bert import BertForSyntacticParsing
from .configuration_bert import BertConfig
from .load_data_utils import ConlluDataset, Args, dep_parse_data_collator
from .path_utils import ModelFolderHandler
from .metrics_utils import dep_parse_metrics_computor

if not is_transformers_available():
    raise BaseException("The `transformers` package was not found")
