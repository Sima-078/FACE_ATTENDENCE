import cv2
import os
import csv
from datetime import datetime

# -----------------------------
# Load trained model
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.read("trainer/trainer.yml")


# -----------------------------
# Face detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# -----------------------------
# Load students
# -----------------------------
students = {}

if os.path.exists("students.csv"):

    with open("students.csv", "r", newline="") as file:

        reader = csv.reader(file)
        next(reader, None)

        for row in reader:

            if len(row) >= 2:

                roll = row[0].strip()
                name = row[1].strip()

                students[roll] = name


# -----------------------------
# Load ID -> Roll mapping
# -----------------------------
id_to_roll = {}

if os.path.exists("trainer/student_names.txt"):

    with open("trainer/student_names.txt", "r") as file:

        for line in file:

            data = line.strip().split(",")

            if len(data) == 2:

                student_id = int(data[0])
                roll = data[1].strip()

                id_to_roll[student_id] = roll


if len(students) == 0:

    print("No student registered.")
    exit()


if len(id_to_roll) == 0:

    print("No trained student found.")
    exit()


# -----------------------------
# Attendance file
# -----------------------------
os.makedirs("attendance", exist_ok=True)

attendance_file = "attendance/attendance.csv"


if not os.path.exists(attendance_file):

    with open(
        attendance_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["Roll Number", "Name", "Date", "Time"]
        )


# -----------------------------
# Already marked today
# -----------------------------
today = datetime.now().strftime("%d-%m-%Y")

marked_students = set()


with open(
    attendance_file,
    "r",
    newline=""
) as file:

    reader = csv.reader(file)

    next(reader, None)

    for row in reader:

        if len(row) >= 4:

            roll = row[0]
            date = row[2]

            if date == today:

                marked_students.add(roll)


# -----------------------------
# Camera
# -----------------------------
camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("Camera could not be opened.")
    exit()


print()
print("Camera started.")
print("Look at the camera.")
print("Press Q to stop.")
print()


# -----------------------------
# Recognition loop
# -----------------------------
while True:

    ret, frame = camera.read()

    if not ret:

        print("Could not read camera.")
        break


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )


    for (x, y, w, h) in faces:

        # Get face only
        face = gray[y:y+h, x:x+w]


        # Same preprocessing as training
        face = cv2.resize(
            face,
            (200, 200)
        )

        face = cv2.equalizeHist(face)


        # Recognize
        student_id, confidence = recognizer.predict(face)


        # Lower confidence = better match
        if (
            student_id in id_to_roll
            and confidence < 55
        ):

            roll = id_to_roll[student_id]


            if roll in students:

                name = students[roll]


                # Green box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    name,
                    (x, y-35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    roll,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


                # Attendance
                if roll not in marked_students:

                    now = datetime.now()

                    date = now.strftime("%d-%m-%Y")
                    time = now.strftime("%H:%M:%S")


                    with open(
                        attendance_file,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow(
                            [
                                roll,
                                name,
                                date,
                                time
                            ]
                        )


                    marked_students.add(roll)


                    print(
                        f"Attendance marked: "
                        f"{roll} - {name} - "
                        f"{date} {time}"
                    )


                else:

                    cv2.putText(
                        frame,
                        "Attendance Already Marked",
                        (x, y+h+30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )


        else:

            # Unknown face
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )


            cv2.putText(
                frame,
                "Unknown",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


    cv2.imshow(
        "Face Recognition Attendance",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# -----------------------------
# Stop camera
# -----------------------------
camera.release()

cv2.destroyAllWindows()


print()
print("Attendance system stopped.")
print(
    "Attendance saved in:",
    attendance_file
)