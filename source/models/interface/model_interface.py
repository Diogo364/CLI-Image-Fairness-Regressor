from abc import ABC
from source.logging import log


class ModelInterface(ABC):
    def __init__(self,
                model_path: str,
                model_loader: object):
        super().__init__()

        log(f'Loading model from {model_path}', 'D')
        self._model_path = model_path
        self.model = model_loader(model_path)

    def __call__(self, image, *args):
        return self.model(image, *args)