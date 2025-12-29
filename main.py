
import cv2
import time
import torch
import pickle
import os
import numpy as np
import pandas as pd
import pyttsx3
import threading
from queue import Queue
from transformers import AutoModelForCausalLM, AutoTokenizer
from facedetector import FaceMeshDetector

debate_topic = "AI taking over jobs"

positions_debate = {
    0: "lightly disagrees with the user statement",
    1: "against the user statement",
    2: "completely against the user statements (extremist position)"
}

moods_debate = {
    0: "neutral person",
    1: "a person who uses very fancy, sophisticated and big words all the time",
    2: "a person that explains things like a baby",
    3: "a grumpy teenager that uses informal internet slang while talking"
}

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


EMOTION_MAP = {
    "angry": "frustrated",
    "happy": "engaged",
    "sad": "frustrated",
    "shocked": "confused",
    "neutral": "neutral"
}


USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
DTYPE = torch.float16 if USE_GPU else torch.float32

print(f"Device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if USE_GPU else None
)

if not USE_GPU:
    model.to(DEVICE)

print("Debate model loaded!")

print("Loading emotion detection model")

with open("model.pkl", "rb") as f:
    emotion_model = pickle.load(f)

selected_indices = (
        list(range(61, 89)) +
        list(range(55, 66)) +
        list(range(285, 296)) +
        [33, 133, 160, 159, 158, 144, 153, 154, 155] +
        [263, 362, 387, 386, 385, 373, 380, 381, 382]
)

extra_cols = ["mouth_open", "mouth_width", "left_eye_height",
              "right_eye_height", "left_brow_eye", "right_brow_eye"]

columns = ["Class"]
columns += [f"x{val}" for val in range(1, len(selected_indices) + 1)]
columns += [f"y{val}" for val in range(1, len(selected_indices) + 1)]
columns += extra_cols

print("✅ Emotion model loaded!")


class DebateEngine:
    def __init__(self, topic, position, mood):
        self.topic = topic
        self.position = position
        self.mood = mood
        self.history = []

    def generate_response(self, user_argument, emotion="neutral"):
        system_prompt = self.build_system_prompt(emotion)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_argument})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        self.history.append({"role": "user", "content": user_argument})
        self.history.append({"role": "assistant", "content": response})

        if len(self.history) > 6:
            self.history = self.history[-6:]

        return response

    def build_system_prompt(self, emotion):
        emotion_instructions = {
            "frustrated": "The user seems frustrated. Acknowledge their points respectfully and be more measured.",
            "engaged": "The user is engaged. Push harder on logical reasoning.",
            "confused": "The user seems confused. Clarify and simplify.",
            "neutral": "Proceed with balanced, logical debate."
        }

        return f"""
You are debating: {self.topic}
Your position: {self.position}
Your personality: {self.mood}

Rules:
- Stay strictly on topic
- Make clear, logical arguments
- Respond directly to user's points
- Keep responses under 100 words
- Be respectful but firm

{emotion_instructions.get(emotion, emotion_instructions["neutral"])}

Don't repeat previous arguments.
"""


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
    return [mouth_open, mouth_width, left_eye_height, right_eye_height,
            left_brow_eye, right_brow_eye]


def normalize_landmarks(face, image_shape):
    h, w, _ = image_shape
    points = np.array([(lm[0] * w, lm[1] * h) for lm in face])
    nose = points[1]
    points = points - nose
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points = points / max_dist
    points = np.round(points, 4)
    return points.flatten()


def detect_emotion(faces, img_shape):
    if len(faces) == 0:
        return "neutral", 0.0

    norm_landmarks = normalize_landmarks(faces[0], img_shape)

    selected_features = []
    for i in selected_indices:
        selected_features.append(norm_landmarks[2 * i])
        selected_features.append(norm_landmarks[2 * i + 1])

    extra = extra_features(norm_landmarks.reshape(-1, 2))
    input_data = pd.DataFrame([selected_features + extra], columns=columns[1:])

    probabilities = emotion_model.predict_proba(input_data)
    confidence = np.max(probabilities)
    result = emotion_model.predict(input_data)[0]

    return (result if confidence > 0.60 else "neutral"), confidence


class EmotionTracker:
    def __init__(self):
        self.current_emotion = "neutral"
        self.emotion_queue = Queue()
        self.running = True
        self.detector = FaceMeshDetector()

    def run(self):
        cap = cv2.VideoCapture(0)
        pTime = 0

        while self.running:
            success, img = cap.read()
            if not success:
                continue

            img, faces = self.detector.find_face(img, True)

            if len(faces) != 0:
                emotion, confidence = detect_emotion(faces, img.shape)
                self.current_emotion = emotion

                cTime = time.time()
                fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
                pTime = cTime

                cv2.putText(img, f"Emotion: {emotion} ({confidence:.2f})",
                            (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            cv2.imshow("Emotion Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_debate_emotion(self):
        return EMOTION_MAP.get(self.current_emotion, "neutral")

    def stop(self):
        self.running = False


engine = pyttsx3.init()
engine.setProperty("rate", 170)


def speak(text):
    engine.say(text)
    engine.runAndWait()


def main():
    debate = DebateEngine(
        topic=debate_topic,
        position=positions_debate[2],
        mood=moods_debate[1]
    )


    emotion_tracker = EmotionTracker()

    emotion_thread = threading.Thread(target=emotion_tracker.run, daemon=True)
    emotion_thread.start()

    print("\n" + "=" * 50)
    print(f"Topic: {debate_topic}")
    print("Type 'exit' or q to quit")
    print("=" * 50 +"\n")

    try:
        while emotion_tracker.running:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in {"exit", "quit", "bye"}:
                print("\nThanks for ur attention")
                break

            if not user_input:
                continue


            debate_emotion = emotion_tracker.get_debate_emotion()
            print(f"[Detected emotion: {emotion_tracker.current_emotion} → {debate_emotion}]")


            response = debate.generate_response(user_input, emotion=debate_emotion)

            print(f"\nAI: {response}\n")
            speak(response)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nDebate ended.")
    finally:
        emotion_tracker.stop()
        emotion_thread.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()