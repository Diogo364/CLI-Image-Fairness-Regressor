import dlib


def clip_face(image, face_detector, shape_predictor, device='cpu', default_max_size=800, size=300, padding=0.25):
    old_height, old_width, _ = image.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
    
    image = dlib.resize_image(image, rows=new_height, cols=new_width)

    faces = face_detector(image, 1)
    num_faces = len(faces)
    if num_faces == 0:
        return None
    faces_landmarks = dlib.full_object_detections()
    for detection in faces:
        rectangle = detection.rect
        faces_landmarks.append(shape_predictor(image, rectangle))
    clipped_images = dlib.get_face_chips(image, faces_landmarks, size=size, padding=padding)
    return clipped_images[0]
    