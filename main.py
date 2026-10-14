import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import cv2
import os
import subprocess
import sys
import csv
import numpy as np


# ============================================================
# BUILD EXISTING FACE MODEL
# ============================================================

def build_existing_model():

    dataset_path = "dataset"

    if not os.path.exists(dataset_path):
        return None, {}

    folders = []

    for folder in os.listdir(dataset_path):

        folder_path = os.path.join(
            dataset_path,
            folder
        )

        if os.path.isdir(folder_path):
            folders.append(folder)

    folders.sort()

    if len(folders) == 0:
        return None, {}

    # SAME LBPH SETTINGS AS train_model.py
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    faces = []
    ids = []
    id_to_roll = {}

    student_id = 1

    for roll in folders:

        folder_path = os.path.join(
            dataset_path,
            roll
        )

        for image_name in os.listdir(folder_path):

            image_path = os.path.join(
                folder_path,
                image_name
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:

                # SAME PREPROCESSING AS TRAINING
                image = cv2.resize(
                    image,
                    (200, 200)
                )

                image = cv2.equalizeHist(image)

                faces.append(image)
                ids.append(student_id)

        id_to_roll[student_id] = roll

        student_id += 1

    if len(faces) == 0:
        return None, {}

    try:

        recognizer.train(
            faces,
            np.array(ids)
        )

        return recognizer, id_to_roll

    except Exception:

        return None, {}


# ============================================================
# DELETE FOLDER
# ============================================================

def delete_folder(folder):

    if os.path.exists(folder):

        for image_name in os.listdir(folder):

            image_path = os.path.join(
                folder,
                image_name
            )

            try:
                os.remove(image_path)
            except:
                pass

        try:
            os.rmdir(folder)
        except:
            pass


# ============================================================
# REGISTER STUDENT
# ============================================================

def register_student():

    window = tk.Toplevel(root)

    window.title(
        "Register Student"
    )

    window.geometry(
        "450x350"
    )

    window.resizable(
        False,
        False
    )

    tk.Label(
        window,
        text="Register New Student",
        font=("Arial", 20, "bold")
    ).pack(pady=20)

    tk.Label(
        window,
        text="Roll Number",
        font=("Arial", 12)
    ).pack()

    roll_entry = tk.Entry(
        window,
        font=("Arial", 12),
        width=30
    )

    roll_entry.pack(pady=5)

    tk.Label(
        window,
        text="Student Name",
        font=("Arial", 12)
    ).pack(pady=10)

    name_entry = tk.Entry(
        window,
        font=("Arial", 12),
        width=30
    )

    name_entry.pack(pady=5)

    def capture_student():

        roll = roll_entry.get().strip()
        name = name_entry.get().strip()

        # ====================================================
        # CHECK EMPTY INPUT
        # ====================================================

        if roll == "" or name == "":

            messagebox.showwarning(
                "Missing Information",
                "Please enter Roll Number and Student Name."
            )

            return

        # ====================================================
        # LOAD STUDENT DETAILS
        # ====================================================

        students_file = "students.csv"

        students = {}

        if os.path.exists(students_file):

            with open(
                students_file,
                "r",
                newline=""
            ) as file:

                reader = csv.reader(file)

                next(reader, None)

                for row in reader:

                    if len(row) >= 2:

                        students[
                            row[0].strip()
                        ] = row[1].strip()

        # ====================================================
        # CHECK ROLL NUMBER
        # ====================================================

        if roll in students:

            messagebox.showwarning(
                "Roll Number Already Exists",
                f"This Roll Number is already registered.\n\n"
                f"Roll Number: {roll}\n"
                f"Name: {students[roll]}"
            )

            return

        # ====================================================
        # BUILD EXISTING FACE MODEL
        # ====================================================

        existing_model, id_to_roll = (
            build_existing_model()
        )

        # ====================================================
        # CAMERA
        # ====================================================

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            messagebox.showerror(
                "Camera Error",
                "Camera could not be opened."
            )

            return

        # ====================================================
        # FACE DETECTOR
        # ====================================================

        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        # ====================================================
        # NEW STUDENT FOLDER
        # ====================================================

        folder = os.path.join(
            "dataset",
            roll
        )

        os.makedirs(
            folder,
            exist_ok=True
        )

        count = 0

        duplicate_roll = None

        # Number of consecutive frames matching
        # an already registered face
        duplicate_hits = 0

        # ====================================================
        # CAPTURE FACE
        # ====================================================

        while True:

            ret, frame = camera.read()

            if not ret:
                break

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5
            )

            for (x, y, w, h) in faces:

                face = gray[
                    y:y+h,
                    x:x+w
                ]

                # =================================================
                # SAME PREPROCESSING AS TRAINING
                # =================================================

                face = cv2.resize(
                    face,
                    (200, 200)
                )

                face = cv2.equalizeHist(
                    face
                )

                # =================================================
                # CHECK IF FACE ALREADY REGISTERED
                # =================================================

                if existing_model is not None:

                    try:

                        student_id, confidence = (
                            existing_model.predict(face)
                        )

                        # IMPORTANT:
                        # Lower LBPH distance = better match.
                        #
                        # If this face matches an existing student
                        # strongly for several consecutive frames,
                        # registration is rejected.

                        if (
                            student_id in id_to_roll
                            and confidence < 55
                        ):

                            duplicate_roll = (
                                id_to_roll[student_id]
                            )

                            duplicate_hits += 1

                        else:

                            duplicate_hits = 0

                        # Require 5 consecutive matching frames
                        if duplicate_hits >= 5:

                            break

                    except Exception:

                        duplicate_hits = 0

                # =================================================
                # IF DUPLICATE FOUND, DO NOT SAVE
                # =================================================

                if duplicate_roll is not None:
                    break

                # =================================================
                # SAVE NEW FACE
                # =================================================

                if count < 40:

                    count += 1

                    image_path = os.path.join(
                        folder,
                        f"{count}.jpg"
                    )

                    cv2.imwrite(
                        image_path,
                        face
                    )

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x+w, y+h),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Images: {count}/40",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            # ====================================================
            # DUPLICATE FACE FOUND
            # ====================================================

            if duplicate_roll is not None:

                cv2.putText(
                    frame,
                    "FACE ALREADY REGISTERED",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "Register Student - Look at Camera",
                    frame
                )

                cv2.waitKey(1500)

                break

            # ====================================================
            # SHOW CAMERA
            # ====================================================

            cv2.imshow(
                "Register Student - Look at Camera",
                frame
            )

            key = cv2.waitKey(100)

            if key == ord("q"):
                break

            if count >= 40:
                break

        # ====================================================
        # CLOSE CAMERA
        # ====================================================

        camera.release()

        cv2.destroyAllWindows()

        # ====================================================
        # DUPLICATE RESULT
        # ====================================================

        if duplicate_roll is not None:

            # VERY IMPORTANT:
            # Delete all newly captured images.
            delete_folder(folder)

            messagebox.showerror(
                "Registration Failed",
                f"This face is already registered.\n\n"
                f"Existing Roll Number: {duplicate_roll}\n\n"
                f"You cannot register the same face "
                f"with another Roll Number or Name."
            )

            return

        # ====================================================
        # NO FACE / NO IMAGES
        # ====================================================

        if count == 0:

            delete_folder(folder)

            messagebox.showerror(
                "Registration Failed",
                "No face was detected.\n"
                "Please try again."
            )

            return

        # ====================================================
        # SAVE STUDENT ONLY AFTER SUCCESSFUL FACE CHECK
        # ====================================================

        students[roll] = name

        with open(
            students_file,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "Roll Number",
                    "Name"
                ]
            )

            for student_roll, student_name in students.items():

                writer.writerow(
                    [
                        student_roll,
                        student_name
                    ]
                )

        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        messagebox.showinfo(
            "Registration Complete",
            f"Student registered successfully!\n\n"
            f"Roll Number: {roll}\n"
            f"Name: {name}\n"
            f"Images Captured: {count}\n\n"
            f"Now train the model."
        )

        window.destroy()

    # ========================================================
    # START FACE CAPTURE BUTTON
    # ========================================================

    tk.Button(
        window,
        text="Start Face Capture",
        font=("Arial", 13, "bold"),
        width=22,
        height=2,
        command=capture_student
    ).pack(pady=25)


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    try:

        result = subprocess.run(
            [sys.executable, "train_model.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:

            messagebox.showinfo(
                "Training Completed",
                "Face recognition model trained successfully."
            )

        else:

            messagebox.showerror(
                "Training Error",
                result.stderr
            )

    except Exception as error:

        messagebox.showerror(
            "Error",
            str(error)
        )


# ============================================================
# TAKE ATTENDANCE
# ============================================================

def take_attendance():

    try:

        subprocess.Popen(
            [sys.executable, "attendance.py"]
        )

    except Exception as error:

        messagebox.showerror(
            "Error",
            str(error)
        )


# ============================================================
# VIEW ATTENDANCE
# ============================================================

def view_attendance():

    attendance_file = os.path.join(
        "attendance",
        "attendance.csv"
    )

    if not os.path.exists(attendance_file):

        messagebox.showwarning(
            "No Attendance",
            "No attendance record found."
        )

        return

    window = tk.Toplevel(root)

    window.title(
        "Attendance Report"
    )

    window.geometry(
        "750x450"
    )

    window.resizable(
        False,
        False
    )

    tk.Label(
        window,
        text="Attendance Report",
        font=("Arial", 20, "bold")
    ).pack(pady=15)

    columns = (
        "Roll Number",
        "Name",
        "Date",
        "Time"
    )

    table = ttk.Treeview(
        window,
        columns=columns,
        show="headings",
        height=14
    )

    for column in columns:

        table.heading(
            column,
            text=column
        )

    table.column(
        "Roll Number",
        width=150,
        anchor="center"
    )

    table.column(
        "Name",
        width=200,
        anchor="center"
    )

    table.column(
        "Date",
        width=150,
        anchor="center"
    )

    table.column(
        "Time",
        width=150,
        anchor="center"
    )

    with open(
        attendance_file,
        "r"
    ) as file:

        reader = csv.reader(file)

        next(reader, None)

        for row in reader:

            if len(row) == 4:

                table.insert(
                    "",
                    "end",
                    values=row
                )

    table.pack(
        padx=20,
        pady=10
    )

    tk.Button(
        window,
        text="Close",
        font=("Arial", 12),
        width=15,
        command=window.destroy
    ).pack(pady=15)


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "Face Recognition Attendance System"
)

root.geometry(
    "700x550"
)

root.resizable(
    False,
    False
)

tk.Label(
    root,
    text="Face Recognition Attendance System",
    font=("Arial", 22, "bold")
).pack(pady=30)

tk.Label(
    root,
    text="Automated Student Attendance System",
    font=("Arial", 13)
).pack(pady=5)


# ============================================================
# REGISTER BUTTON
# ============================================================

tk.Button(
    root,
    text="Register Student",
    font=("Arial", 14),
    width=28,
    height=2,
    command=register_student
).pack(pady=12)


# ============================================================
# TRAIN BUTTON
# ============================================================

tk.Button(
    root,
    text="Train Face Recognition Model",
    font=("Arial", 14),
    width=28,
    height=2,
    command=train_model
).pack(pady=12)


# ============================================================
# ATTENDANCE BUTTON
# ============================================================

tk.Button(
    root,
    text="Take Attendance",
    font=("Arial", 14),
    width=28,
    height=2,
    command=take_attendance
).pack(pady=12)


# ============================================================
# VIEW ATTENDANCE BUTTON
# ============================================================

tk.Button(
    root,
    text="View Attendance",
    font=("Arial", 14),
    width=28,
    height=2,
    command=view_attendance
).pack(pady=12)


# ============================================================
# EXIT BUTTON
# ============================================================

tk.Button(
    root,
    text="Exit",
    font=("Arial", 12),
    width=15,
    command=root.destroy
).pack(pady=15)


# ============================================================
# START APPLICATION
# ============================================================

root.mainloop()