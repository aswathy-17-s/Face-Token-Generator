import cv2
import face_recognition
import os
import time
import random
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Ensure you have the correct model file

# Create folders to save known and unknown faces
os.makedirs('Known_Faces', exist_ok=True)
os.makedirs('Unknown_Faces', exist_ok=True)

# Function to load known faces from a directory
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
known_faces_dir = "Known_Faces"
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Start video capture
video_capture = cv2.VideoCapture("11.mp4")

def generate_token(name):
    dal_qty = round(random.uniform(1, 10), 2)
    rice_qty = round(random.uniform(1, 10), 2)
    wheat_qty = round(random.uniform(1, 10), 2)
    sugar_qty = round(random.uniform(1, 10), 2)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    token_details = (f"Token for {name}\nTime: {current_time}\nItems:\n"
                     f"Dal: {dal_qty} kg\nRice: {rice_qty} kg\nWheat: {wheat_qty} kg\nSugar: {sugar_qty} kg")
    os.makedirs('Tokens', exist_ok=True)
    token_filename = f"Tokens/{name}_token.txt"
    with open(token_filename, 'w') as file:
        file.write(token_details)
    print(f"Token saved for {name}: {token_filename}")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Perform YOLOv8 person detection
    results = yolo_model(frame)
    person_count = 0

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) == 0:  # Class ID 0 corresponds to 'person'
                person_count += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert frame for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            generate_token(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"Persons detected: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('Person Detection & Counting', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()