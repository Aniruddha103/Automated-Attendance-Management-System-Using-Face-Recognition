import face_recognition
import cv2

# Path to your image
image_path = "C:/Users/Aniruddha/OneDrive/Pictures/Camera Roll/grp_crop.jpg"

# Load the image
image = face_recognition.load_image_file(image_path)

# Detect all face locations
face_locations = face_recognition.face_locations(image)

print(f"Found {len(face_locations)} face(s) in this image.")

# Optional: draw rectangles around faces using OpenCV
image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert RGB -> BGR for OpenCV

for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)

# Show the image with faces
cv2.imshow("Faces Detected", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
