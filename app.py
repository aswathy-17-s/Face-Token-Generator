import streamlit as st
import cv2
import face_recognition
import os
import time
import random
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Create necessary directories
os.makedirs('Known_Faces', exist_ok=True)
os.makedirs('Unknown_Faces', exist_ok=True)
os.makedirs('Tokens', exist_ok=True)

# Function to load known faces
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(known_faces_dir):
        person_folder = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    known_image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(known_image)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
    return known_face_encodings, known_face_names

# Load known faces
st.markdown("<h1 style='text-align: center; font-family: Times New Roman;'>🔍 FACE & PERSON DETECTION SYSTEM</h1>", unsafe_allow_html=True)
st.write("""
    🎯 **Welcome to the Face & Person Detection System!**
    
    This advanced AI-based system detects faces and counts the number of people in real-time. 
    Whether you want to generate food distribution tokens based on **Face Recognition** or **Person Count**, 
    our system makes it efficient, accurate, and hassle-free. 🏆

    🚀 **Key Features:**
    - **Real-time Detection**: Detect and recognize individuals instantly.
    - **Face-Based Token Generation**: Assign tokens to known individuals.
    - **Person Count-Based Tokens**: Generate group-based tokens automatically.
    - **Advanced AI Models**: Powered by **YOLOv8** for high-precision person detection.
    - **Beautiful UI/UX**: Enjoy a seamless and interactive experience! 🌟

    📌 **Start exploring and automate your detection needs effortlessly!** 🔥
""")

known_faces_dir = "Known_Faces"
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Function to generate tokens
def generate_token(name, person_count, method):
    dal_qty = round(random.uniform(1, 10) * person_count, 2)
    rice_qty = round(random.uniform(1, 10) * person_count, 2)
    wheat_qty = round(random.uniform(1, 10) * person_count, 2)
    sugar_qty = round(random.uniform(1, 10) * person_count, 2)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    token_details = (f"📝 TOKEN FOR {name}\n⏰ TIME: {current_time}\n🔹 METHOD: {method}\n"
                     f"🍚 ITEMS:\n👉 DAL: {dal_qty} KG\n👉 RICE: {rice_qty} KG\n"
                     f"👉 WHEAT: {wheat_qty} KG\n👉 SUGAR: {sugar_qty} KG")
    
    token_filename = f"Tokens/{name}_token.txt"
    with open(token_filename, 'w', encoding='utf-8') as file:
        file.write(token_details)
    
    st.success(f"✅ TOKEN GENERATED FOR {name}!")
    st.text(token_details)

# Sidebar for user input
method = st.sidebar.radio("🎯 SELECT TOKEN GENERATION METHOD", ["FACE RECOGNITION", "PERSON COUNT"])

# Start video capture
video_capture = cv2.VideoCapture(0)
stframe = st.empty()

run_detection = st.button("🚀 START DETECTION")
stop_detection = st.button("🛑 STOP DETECTION")

if run_detection:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("⚠️ ERROR: UNABLE TO CAPTURE VIDEO.")
            break
        
        results = yolo_model(frame)
        person_count = 0

        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                if int(cls) == 0:  # Person detection
                    person_count += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, "👤 PERSON", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if method == "FACE RECOGNITION":
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "UNKNOWN"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    generate_token(name, person_count, "FACE RECOGNITION")
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 3)
                cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        elif method == "PERSON COUNT":
            generate_token("GROUP", person_count, "PERSON COUNT BASED")

        cv2.putText(frame, f"👥 PERSONS DETECTED: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        stframe.image(frame, channels="BGR", use_column_width=True)

        if stop_detection:
            break

video_capture.release()
st.success("🎉 DETECTION STOPPED SUCCESSFULLY!")
