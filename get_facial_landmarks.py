import os,sys,dlib

model = "shape_predictor_68_face_landmarks.dat"

def load_landmarks_model(model):
    try:
        landmark_model = model
        predictor_path = os.path.join(os.getcwd(), landmark_model)
        predictor = dlib.shape_predictor(predictor_path)
    except IOError:
        print("Please put facial landmark model in current folder !")
    return predictor

def detect_image(predictor,image_path):
    detector =  dlib.get_frontal_face_detector()
    image = dlib.load_rgb_image(image_path)
    faces = detector(image, 1)

    for k, d in enumerate(faces):
        # print("Detection {}: {}".format(k, d))
        markers = predictor(image, d)
        landmarks = markers.parts()
    return landmarks

if __name__ == "__main__" :
    predictor = load_landmarks_model(model)

    image = "./donald_trump.jpg"
    landmarks = detect_image(predictor,image)
    print(landmarks)