# Smart Attendance System Using Face Recognition

A desktop-based automated attendance management system using OpenCV and Python. This application captures face samples, trains an LBPH (Local Binary Patterns Histograms) face recognition model, and marks student attendance in real time.

---

## 📌 Features

- **Graphical User Interface (GUI)**: Intuitive Tkinter-based desktop interface.
- **Student Registration & Face Capture**: Automated face detection using Haar Cascade and sample image collection.
- **Model Training**: Accurate face recognition powered by OpenCV's LBPH Face Recognizer.
- **Real-Time Attendance Marking**: Live webcam feed detects recognized faces, verifies identity, and records roll number, name, date, and timestamp.
- **Attendance Records**: Exports attendance logs directly to CSV for record-keeping.

---

## 🛠️ Tech Stack & Requirements

- **Python 3.8+**
- **OpenCV (`opencv-contrib-python`)**: For Haar Cascades and LBPH face recognition.
- **NumPy**: For array and image matrix manipulation.
- **Tkinter**: Standard Python library for GUI.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```text
├── dataset/             # Captured face images for each student (excluded from git)
├── trainer/             # Trained LBPH model file (.yml) (excluded from git)
├── attendance/          # Generated attendance logs (.csv)
├── students.csv         # Registered students database (Roll Number, Name)
├── main.py              # Main GUI application
├── capture_faces.py     # Face dataset collection module
├── train_model.py       # Model training script
├── attendance.py        # Real-time attendance recognition engine
├── requirements.txt     # Python package requirements
└── README.md            # Project documentation
```

---

## 📖 How to Use

1. **Launch the Application**:
   ```bash
   python main.py
   ```
2. **Register a Student**:
   - Enter student Roll Number and Name.
   - Click **Capture Faces** to record face sample frames from your webcam.
3. **Train Model**:
   - Click **Train Model** to generate/update the LBPH recognition weights in `trainer/trainer.yml`.
4. **Mark Attendance**:
   - Click **Take Attendance** to launch the camera recognition stream.
   - Once recognized, attendance is automatically recorded in `attendance/attendance.csv`.
