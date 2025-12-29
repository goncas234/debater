import csv
from facedetector import FaceMeshDetector
import time
import cv2
import numpy as np

selected_indices = (
    list(range(61, 89)) +                                       # Mouth
    list(range(55, 66)) +                                       # Left eyebrow
    list(range(285, 296)) +                                     # Right eyebrow
    [33, 133, 160, 159, 158, 144, 153, 154, 155] +              # Left eye
    [263, 362, 387, 386, 385, 373, 380, 381, 382]               # Right eye
)

class_name = "neutral"

extra_cols = ["mouth_open","mouth_width","left_eye_height","right_eye_height","left_brow_eye","right_brow_eye"]
columns = ["Class"] + [f"x{i}" for i in range(1, len(selected_indices)+1)] \
          + [f"y{i}" for i in range(1, len(selected_indices)+1)] \
          + extra_cols


#with open("data.csv","w",newline="") as file:
#    csv_writer = csv.writer(file,delimiter=",")
#    csv_writer.writerow(columns)

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

    return points.flatten().tolist()


def main():
    print(len(selected_indices))
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    with open("data.csv","a",newline="") as f:
        csv_writer = csv.writer(f, delimiter=",")
        while True:
            success, img = cap.read()
            img, faces = detector.find_face(img,True)

            if len(faces) != 0:
                #print(faces[0])
                #print(len(faces[0]))

                norm_landmarks = normalize_landmarks(faces[0], img.shape)

                selected_features = []
                for i in selected_indices:
                    selected_features.append(norm_landmarks[2 * i])
                    selected_features.append(norm_landmarks[2 * i + 1])

                extra = extra_features(np.array(norm_landmarks))
                row = [class_name] + selected_features + extra
                csv_writer.writerow(row)
                time.sleep(0.1)

            cv2.imshow("Image", img)
            cv2.waitKey(1)



if __name__ == "__main__":
    main()