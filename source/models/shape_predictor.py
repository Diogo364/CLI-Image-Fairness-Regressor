import dlib
from source.models.interface.model_interface import ModelInterface

class ShapePredictor(ModelInterface):
    def __init__(self,
                model_path='/assets/dlib_models/shape_predictor_5_face_landmarks.dat',
                model_loader=dlib.shape_predictor):
        super().__init__(model_path, model_loader)

    
    def __call__(self, image, rectangle):
        return super().__call__(image, rectangle)

