import cv2
import os

# Student details
student_id = input("Enter Roll Number: ")
student_name = input("Enter Student Name: ")

# Create student folder
student_folder = os.path.join("dataset", student_id)

if not os.path.exists(student_folder):
    os.makedirs(student_folder)

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start camera
camera = cv2.VideoCapture(0)

count = 0

print("\nCamera started.")
print("Look at the camera...")
print("Press Q to stop.\n")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Camera could not be opened.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        count += 1

        # Save face image
        file_name = os.path.join(
            student_folder,
            f"{student_id}_{count}.jpg"
        )

        cv2.imwrite(file_name, gray[y:y+h, x:x+w])

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}/30",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Capture - Press Q to Quit", frame)

    # Stop after 30 images
    key = cv2.waitKey(100) & 0xff

    if key == ord("q"):
        break

    if count >= 30:
        break

camera.release()
cv2.destroyAllWindows()

print("\nFace capture completed!")
print(f"Images saved in: {student_folder}")