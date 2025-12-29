I built this project a few months ago right after I turned 16. I used Python 3.10, since it was the only version compatible with MediaPipe I had. I really enjoyed creating it, and it was a fun challenge to combine real time emotion detection with an AI debate system. In the future, I plan to update the model I’m using, get better data for the emotions detector and fix the text-to-speech functionality that was working previously on colab.

Project: AI debater that adjusts it's talking based on your facial emotions
This project is an AI debate system that interacts with a user in a surprisingly human way. The core idea is that the AI doesn’t just respond to what you type but it also observes your facial expressions in real time using your webcam. By tracking your face with MediaPipe and analyzing your emotions, the system can tell whether you’re happy, angry, shocked, neutral and on previous versions: sad. With this it adjusts the talking to be more human like. 

I'm planning on maybe building an ui in the future with the functionality of changing the ai's voice, personality and way of talking but for now I only implemented it locally.

My main challenge was creating a system that could maintain the webcam active as long with the llm but claude sugested me this idea of "threading" and it worked just fine.

Instead of trying to collect a massive dataset of facial expressions, I decided to use a simpler approach based on distances between key points on the face. By measuring how far certain points on the mouth, eyes, and eyebrows move relative to each other, I could extract meaningful features for emotion detection. :)

Hope u like it and if everythings broken pls tell me as this is one of my first github worth projects


