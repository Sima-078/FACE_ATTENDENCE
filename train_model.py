import cv2
import os
import numpy as np

dataset_path = "dataset"
trainer_path = "trainer"

os.makedirs(trainer_path, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

student_folders = []

for folder_name in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder_name)

    if os.path.isdir(folder_path):
        student_folders.append(folder_name)

student_folders.sort()

faces = []
ids = []

for student_index, roll in enumerate(student_folders, start=1):

    print("Reading:", roll)

    student_folder = os.path.join(dataset_path, roll)

    for image_name in os.listdir(student_folder):

        image_path = os.path.join(student_folder, image_name)

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        # Same size for every training image
        image = cv2.resize(image, (200, 200))

        # Improve contrast
        image = cv2.equalizeHist(image)

        faces.append(image)
        ids.append(student_index)

print()
print("Training started...")
print("Number of face images:", len(faces))

if len(faces) == 0:
    print("No face images found!")
    exit()

recognizer.train(
    faces,
    np.array(ids)
)

recognizer.write(
    "trainer/trainer.yml"
)

with open(
    "trainer/student_names.txt",
    "w"
) as file:

    for student_index, roll in enumerate(
        student_folders,
        start=1
    ):

        file.write(
            f"{student_index},{roll}\n"
        )

print()
print("Training completed successfully!")
print("Model saved at: trainer/trainer.yml")
print("Student mapping saved at: trainer/student_names.txt")