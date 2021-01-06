import os


class ModelFolderHandler: 
    def __init__(self, path_root, model_name):
        if not os.path.isdir(path_root):
            raise BaseException("The folder was not found : " + path_root)

        self.path_models = os.path.join(path_root, "models")
        self._create_path_if_not_exist(self.path_models)

        self.path_model = os.path.join(self.path_models, model_name)
        self._create_path_if_not_exist(self.path_model)

        self.training_args = os.path.join(self.path_model, "training_args")
        
        self.saved_model = os.path.join(self.path_models, "saved_model")

        self.annotation_schema = os.path.join(path_root, "annotation_schema.json")

        self.train_folder = os.path.join(path_root, "train")
        self.eval_folder = os.path.join(path_root, "eval")
        
    def _create_path_if_not_exist(self, path) -> None:
        if not os.path.isdir(path):
            os.mkdir(path)
    