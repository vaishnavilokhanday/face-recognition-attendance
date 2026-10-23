import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import csv
import tkinter as tk
from PIL import Image, ImageTk

# Load known faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    base_dir = 'known_faces'
    for person_name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        for image_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image_name)
            print(f"[INFO] Loading: {img_path}")
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
                print(f"[✓] Face encoded for {person_name}")
            else:
                print(f"[X] No face found in {img_path}")

# Attendance marking
def mark_attendance(name):
    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')

    file_exists = os.path.isfile('attendance.csv')
    with open('attendance.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time'])
        writer.writerow([name, date_str, time_str])
        print(f"[ATTENDANCE] {name} marked at {time_str} on {date_str}")

# GUI and Camera logic
class AttendanceApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Recognition Attendance")
        self.video_capture = cv2.VideoCapture(0)

        self.frame_label = tk.Label(window)
        self.frame_label.pack()

        self.button = tk.Button(window, text="Mark Attendance", command=self.mark_face)
        self.button.pack(pady=10)

        self.current_name = None
        self.running = True
        self.process_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def process_frame(self):
        if not self.running:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            return

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        self.current_name = None

        names = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match = np.argmin(face_distances) if face_distances.size else None

            name = "Unknown"
            if best_match is not None and matches[best_match]:
                name = known_face_names[best_match]
                self.current_name = name  # For attendance marking
            names.append((name, face_location))

        # Draw boxes and labels on the original frame (not the small one)
        for name, (top, right, bottom, left) in names:
            # Scale back up face locations since the frame we detected on was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 2)

        # Convert the frame to RGB and then to ImageTk format for Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.frame_label.imgtk = imgtk
        self.frame_label.configure(image=imgtk)

        self.window.after(10, self.process_frame)

    def mark_face(self):
        if self.current_name:
            mark_attendance(self.current_name)

    def on_closing(self):
        self.running = False
        self.video_capture.release()
        self.window.destroy()

# Main execution
if __name__ == "__main__":
    load_known_faces()
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
