# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import cv2
import numpy as np
import face_recognition
from datetime import datetime, time
import mysql.connector
import os
import time as t

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Attendance System ------------------
images_dir = "C:/Users/Aniruddha/OneDrive/"

images_path = {}
student_ids = {}

for filename in os.listdir(images_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        roll_no = os.path.splitext(filename)[0]
        images_path[roll_no] = os.path.join(images_dir, filename)
        student_ids[roll_no] = int(roll_no)

marked_students = set()
running = False  # global flag for camera thread

# Define class schedule
CLASS_START = time(21, 22)  # 07:51 PM
CLASS_END = time(21, 31)    # 08:01 PM

# ------------------ Database Functions ------------------
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="",
            port=3306,
            user="root",
            password="",
            database="student"
        )

        if connection.is_connected():
            print("Connected to MySQL database")
        return connection
    except mysql.connector.Error as e:
        print(f"Error: {e}")
        return None

def insert_student(student_id, roll_no):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            query = "INSERT INTO students (id, name, timestamp) VALUES (%s, %s, %s)"
            cursor.execute(query, (student_id, roll_no, now))
            connection.commit()
            print(f"Inserted {roll_no} (ID: {student_id}) at {now}")
        except mysql.connector.Error as e:
            print(f"Error: {e}")
        finally:
            cursor.close()
            connection.close()

def mark_attendance(roll_no):
    if roll_no in student_ids and roll_no not in marked_students:
        student_id = student_ids[roll_no]
        insert_student(student_id, roll_no)
        marked_students.add(roll_no)

def load_known_faces_and_names():
    known_face_encodings = []
    known_face_names = []

    for roll_no, path in images_path.items():
        if not os.path.exists(path):
            continue
        
        # Load image
        image = face_recognition.load_image_file(path)
        
        # Preprocess: convert to RGB and enhance contrast/brightness
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=20)  # brighten and increase contrast
        
        # Create face encodings
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(roll_no)

    return known_face_encodings, known_face_names


# ------------------ Camera Thread ------------------
def run_camera():
    global running
    known_face_encodings, known_face_names = load_known_faces_and_names()
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Cannot access camera")
        running = False
        return

    while running:
        ret, frame = video_capture.read()
        if not ret:
            continue
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)  # changed from 0.25 to 0.5
        face_locations = face_recognition.face_locations(small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations, num_jitters=3)
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.42)  
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) == 0:
                continue   
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                roll_no = known_face_names[best_match_index]
                mark_attendance(roll_no)
                top, right, bottom, left = [v*2 for v in face_location]  # adjusted for fx=0.5
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, roll_no, (left, top-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    running = False


# ------------------ Attendance Scheduler ------------------
def attendance_scheduler():
    global running
    while True:
        now = datetime.now().time()
        if not running and CLASS_START <= now < CLASS_END:
            # Start camera
            running = True
            thread = threading.Thread(target=run_camera, daemon=True)
            thread.start()
            print(f"Class started at {now}, camera running...")
        elif running and now >= CLASS_END:
            # Stop camera
            running = False
            print(f"Class ended at {now}, camera stopped.")
        t.sleep(10)  # check every 10 seconds

@app.on_event("startup")
def startup_event():
    scheduler_thread = threading.Thread(target=attendance_scheduler, daemon=True)
    scheduler_thread.start()

# ------------------ FastAPI Endpoints ------------------
@app.get("/attendance")
async def get_attendance():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM students ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return rows
        except mysql.connector.Error as e:
            return {"error": str(e)}
        finally:
            cursor.close()
            connection.close()
    return {"error": "Database connection failed"}