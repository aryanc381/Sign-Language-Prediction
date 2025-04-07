import requests
import pyttsx3
import time
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Actions and model
actions = np.array(['iloveyou', 'thankyou', 'hi'])
label_map = {label:num for num, label in enumerate(actions)}
model = tf.keras.models.load_model('action350.h5')

# Parameters
colors = [(245,117,16), (117,245,16), (16,117,245)]
sequence = []
sentence = []
threshold = 0.8
detected_words = []  # Store all detected words
last_detection_time = time.time()  # Initialize timer

cap = cv2.VideoCapture(0)

# Run Mediapipe holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        current_time = time.time()
        if len(sequence) == 30 and current_time - last_detection_time >= 3:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            prediction = actions[np.argmax(res)]
            print(prediction)

            if res[np.argmax(res)] > threshold:
                if len(sentence) == 0 or prediction != sentence[-1]:
                    sentence.append(prediction)
                    detected_words.append(prediction)  # Save to final list
                last_detection_time = current_time

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Final output
final_sentence = ' '.join(detected_words)
print("\n✅ Final Detected Words List:")
print(detected_words)
print("\n📝 Final Sentence:")
print(final_sentence)


def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)      # Speed of speech
        engine.setProperty('volume', 1.0)    # Volume (0.0 to 1.0)
        
        # You can try selecting a specific voice if needed
        # Uncomment this to try a female voice if available
        # voices = engine.getProperty('voices')
        # for voice in voices:
        #     if "female" in voice.name.lower():
        #         engine.setProperty('voice', voice.id)
        #         break

        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Speech failed:", e)

def correct_grammar_with_deepseek(text):
    prompt = (
        f"Correct the grammar of this sentence and complete it: \"{text}\".\n"
        "Only reply with the corrected sentence. No other explanation, character, key, word, etc. even if you feel you should type something out, dont do it. I just want the corrected sentence, that is it. I dont want you to say, the corrected sentence is blah blah"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-llm",
            "prompt": prompt,
            "stream": False
        }
    )

    if response.status_code == 200:
        corrected = response.json()["response"].strip().strip('"\n ')
        print("Original:", text)
        print("Corrected:", corrected)
        
        time.sleep(0.5)  # Small delay before speaking
        speak_text(corrected)

        return corrected
    else:
        print("Error:", response.status_code)
        return None

correct_grammar_with_deepseek(final_sentence)                                                                                  