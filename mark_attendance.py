import cv2
import numpy as np
import face_recognition
from datetime import datetime
import mysql.connector
import os

# Path to folder containing student images
images_dir = "C:/Users/Aniruddha/OneDrive/Pictures/Camera Roll/"

# Dynamically build images_path and student_ids from folder
images_path = {}
student_ids = {}

for filename in os.listdir(images_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        roll_no = os.path.splitext(filename)[0]  # filename without extension
        images_path[roll_no] = os.path.join(images_dir, filename)
        student_ids[roll_no] = int(roll_no)  # assume filenames are numeric roll numbers

# Keep track of students already marked in this session
marked_students = set()

def create_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
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

def insert_student(student_id, name):
    """Insert attendance with current timestamp."""
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            query = "INSERT INTO students (id, name, timestamp) VALUES (%s, %s, %s)"
            cursor.execute(query, (student_id, name, now))
            connection.commit()
            print(f"Inserted {name} (ID: {student_id}) at {now}")
        except mysql.connector.Error as e:
            print(f"Error: {e}")
        finally:
            cursor.close()
            connection.close()

def mark_attendance(roll_no):
    """Mark attendance only once per session."""
    if roll_no in student_ids and roll_no not in marked_students:
        student_id = student_ids[roll_no]
        insert_student(student_id, roll_no)
        marked_students.add(roll_no)
    elif roll_no in marked_students:
        print(f"{roll_no} already marked in this session.")
    else:
        print(f"Error: ID for {roll_no} not found!")

def load_known_faces_and_names(images_path):
    known_face_encodings = []
    known_face_names = []
    for roll_no, path in images_path.items():
        if not os.path.exists(path):
            print(f"Warning: Image file {path} not found.")
            continue
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(roll_no)
        else:
            print(f"Warning: No face detected in {path}. Skipping.")
    return known_face_encodings, known_face_names

def main():
    known_face_encodings, known_face_names = load_known_faces_and_names(images_path)
    video_capture = cv2.VideoCapture(0)  # change camera index if needed
    if not video_capture.isOpened():
        print("Error: Unable to access camera.")
        return
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) == 0:
                continue
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                roll_no = known_face_names[best_match_index]
                mark_attendance(roll_no)
                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, roll_no, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
