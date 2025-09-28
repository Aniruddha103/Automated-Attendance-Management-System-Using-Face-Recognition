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
images_dir = "C:/Users/Aniruddha/OneDrive/Students/"  # folder with student images
images_path = {}
student_ids = {}
marked_students = set()
lock = threading.Lock()  # Thread-safe access
running = False

# Load student images
for filename in os.listdir(images_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        roll_no = os.path.splitext(filename)[0]
        images_path[roll_no] = os.path.join(images_dir, filename)
        student_ids[roll_no] = int(roll_no)

# Define class schedule
CLASS_START = time(20, 3)  # 08:03 PM
CLASS_END = time(20, 5)    # 08:05 PM

# ------------------ Database Functions ------------------
def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password="",  # set your password
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
    with lock:
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
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(roll_no)
    return known_face_encodings, known_face_names

# ------------------ Camera Thread ------------------
def run_camera():
    global running
    known_face_encodings, known_face_names = load_known_faces_and_names()
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not video_capture.isOpened():
        print("Cannot access camera")
        running = False
        return

    scale = 0.75  # Increase resolution for higher accuracy
    while running:
        ret, frame = video_capture.read()
        if not ret:
            continue

        # Convert to RGB
        small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces using CNN for high accuracy
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) == 0:
                continue
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                roll_no = known_face_names[best_match_index]
                mark_attendance(roll_no)

                # Scale back face locations for display
                top, right, bottom, left = [int(v/scale) for v in face_location]
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
            running = True
            thread = threading.Thread(target=run_camera, daemon=True)
            thread.start()
            print(f"Class started at {now}, camera running...")
        elif running and now >= CLASS_END:
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