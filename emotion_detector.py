
from facedetector import FaceMeshDetector
import time
import cv2
import numpy as np
import pickle
import pandas as pd
import pyttsx3

engine = pyttsx3.init()

normal = "neutral"

selected_indices = (
    list(range(61, 89)) +                                       # Mouth
    list(range(55, 66)) +                                       # Left eyebrow
    list(range(285, 296)) +                                     # Right eyebrow
    [33, 133, 160, 159, 158, 144, 153, 154, 155] +              # Left eye
    [263, 362, 387, 386, 385, 373, 380, 381, 382]               # Right eye
)

with open("model.pkl","rb") as f:
    m = pickle.load(f)


extra_cols = ["mouth_open","mouth_width","left_eye_height","right_eye_height","left_brow_eye","right_brow_eye"]

columns = ["Class"]
columns += [f"x{val}" for val in range(1, len(selected_indices)+1)]
columns += [f"y{val}" for val in range(1, len(selected_indices)+1)]
columns += extra_cols

def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extra_features(points):
    pts = points.reshape(-1, 2)

    mouth_open = euclidean_dist(pts[61], pts[67])
    mouth_width = euclidean_dist(pts[61], pts[291])

    left_eye_height = euclidean_dist(pts[159], pts[145])
    right_eye_height = euclidean_dist(pts[386], pts[374])

    left_brow_eye = euclidean_dist(pts[55], pts[159])
    right_brow_eye = euclidean_dist(pts[285], pts[386])

    return [mouth_open, mouth_width, left_eye_height, right_eye_height, left_brow_eye, right_brow_eye]


def normalize_landmarks(face, image_shape):

    h, w, _ = image_shape
    points = np.array([(lm[0] * w, lm[1] * h) for lm in face])

    nose = points[1]
    points = points - nose

    max_dist = np.max(np.linalg.norm(points, axis=1))
    points = points / max_dist

    points = np.round(points, 4)

    return points.flatten()


def main():


    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.find_face(img,True)

        if len(faces) != 0:
            #print(faces[0])
            #print(len(faces[0]))
            #d.append(class_name)
            norm_landmarks = normalize_landmarks(faces[0], img.shape)

            selected_features = []
            for i in selected_indices:
                selected_features.append(norm_landmarks[2 * i])
                selected_features.append(norm_landmarks[2 * i + 1])


            extra = extra_features(norm_landmarks.reshape(-1, 2))

            input_data = pd.DataFrame([selected_features + extra], columns=columns[1:])

            probabilities = m.predict_proba(input_data)
            confidence = np.max(probabilities)
            result = m.predict(input_data)[0]
            #print(f"Prediction: {result if confidence > 0.50 else normal}, Confidence: {confidence:.2f}")

            time.sleep(0.01)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f"emotion: {result if confidence > 0.60 else normal}",(20,70),cv2.FONT_HERSHEY_PLAIN,
                        3,(0,255,0),3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()