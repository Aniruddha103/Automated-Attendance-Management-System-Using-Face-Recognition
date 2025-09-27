import face_recognition
import cv2
import os

# Path to your image
image_path = "C:/Users/Aniruddha/OneDrive/Pictures/Camera Roll/group_photo2.jpeg"

# Load the image
image = face_recognition.load_image_file(image_path)

# Detect all face locations
face_locations = face_recognition.face_locations(image)

print(f"Found {len(face_locations)} face(s) in this image.")

# Convert to BGR for OpenCV
image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Folder to save detected faces
save_folder = os.path.dirname(image_path)

# Starting numeric name
start_index = 234049

for i, (top, right, bottom, left) in enumerate(face_locations):
    # Extract face
    face_image = image_cv[top:bottom, left:right]
    
    # Create filename
    face_filename = os.path.join(save_folder, f"{start_index + i}.jpg")
    
    # Save face image
    cv2.imwrite(face_filename, face_image)
    print(f"Saved face {i+1} as {face_filename}")

print("All faces saved successfully.")
