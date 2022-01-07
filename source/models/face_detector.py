import dlib
from source.models.interface.model_interface import ModelInterface

class FaceDetector(ModelInterface):
    def __init__(self,
                model_path='/assets/dlib_models/mmod_human_face_detector.dat',
                model_loader=dlib.cnn_face_detection_model_v1):
        super().__init__(model_path, model_loader)
    
    def __call__(self, image, upsample_num_times=0):
        return super().__call__(image, upsample_num_times)
