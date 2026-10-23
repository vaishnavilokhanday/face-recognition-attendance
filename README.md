# Real-Time Attendance Tracker Using Facial Features

A Python-based desktop application that uses facial recognition to identify registered individuals and automatically record their attendance.

## 🚀 Features
- Real-time webcam face detection and recognition.
- Custom graphical interface built with `tkinter`.
- Automatically saves attendance (Name, Date, Time) to a CSV file.
- Prevents duplicate entries and handles unknown faces gracefully.

## 🛠️ Tech Stack
- **Language:** Python
- **Computer Vision:** OpenCV (`cv2`), `face_recognition`
- **GUI:** `tkinter`
- **Data Handling:** `numpy`, `csv`

## 📂 Project Documentation
The full project report and PowerPoint presentation can be found in the `docs/` folder.

## ⚙️ How to Run
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Ensure you have a webcam connected.
4. Add images of people to the known_faces/ folder (create a subfolder for each person).
5. Run the app:
    python main.py
