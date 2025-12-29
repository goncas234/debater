import os
import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import pyttsx3

# =========================
# CONFIGURATION
# =========================

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

# =========================
# DEVICE DETECTION
# =========================

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"
DTYPE = torch.float16 if USE_GPU else torch.float32

print(f"🔥 Device: {DEVICE}")

# =========================
# LOAD MODEL
# =========================

print("Loading model... (first run can take several minutes)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if USE_GPU else None
)

if not USE_GPU:
    model.to(DEVICE)

print("✅ Model loaded!")

# =========================
# DEBATE ENGINE
# =========================

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

# =========================
# TEXT TO SPEECH (LOCAL)
# =========================

#def speak(text, filename="response.mp3"):
#    tts = gTTS(text=text, lang="en")
#    tts.save(filename)

#    if os.name == "nt":  # Windows
#        os.system(f'start {filename}')
#    elif os.name == "posix":
#       os.system(f'afplay {filename} || mpg123 {filename}')


engine = pyttsx3.init()
engine.setProperty("rate", 170)  # speed

def speak(text):
    engine.say(text)
    engine.runAndWait()


# =========================
# INTERACTIVE SESSION
# =========================

debate = DebateEngine(
    topic=debate_topic,
    position=positions_debate[2],
    mood=moods_debate[1]
)

print("\n" + "=" * 50)
print(f"Topic: {debate_topic}")
print("Type 'exit' to quit")
print("=" * 50 + "\n")

try:
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("\nAI: It was a pleasure debating you. Goodbye!")
            break

        if not user_input:
            continue

        response = debate.generate_response(user_input, emotion="engaged")

        print(f"\nAI: {response}\n")
        speak(response)
        print("-" * 50)

except KeyboardInterrupt:
    print("\nDebate ended.")
