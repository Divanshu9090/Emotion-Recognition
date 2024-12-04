import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
try:
    model = load_model("retrained_emotion_recognition_model.h5")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def predict_emotion(frame, facecasc):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        # Preprocess the face for prediction
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # Predict emotion
        prediction = model.predict(roi_gray)[0]
        max_index = int(np.argmax(prediction))
        emotion = emotion_dict[max_index]

        # Display emotion label
        cv2.putText(frame, emotion, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def main():
    st.title("Real-Time Emotion Recognition")

    # Initialize the webcam
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])  # Placeholder for video frames

    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        frame = cv2.flip(frame, 1)  # Mirror the frame
        frame = predict_emotion(frame, facecasc)
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
    st.warning("Webcam stopped.")

if __name__ == "__main__":
    main()
