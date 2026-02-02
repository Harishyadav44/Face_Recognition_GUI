import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import os
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
LABEL_FILE = "labels.txt"
CONFIDENCE_THRESHOLD = 85

os.makedirs(DATASET_DIR, exist_ok=True)

#  GUI 
window = tk.Tk()
window.title("Face Recognition System")
window.geometry("500x300")

#  UTILITY FUNCTIONS 
def get_new_user_id():
    if not os.path.exists(LABEL_FILE):
        return 1
    with open(LABEL_FILE, "r") as f:
        return len(f.readlines()) + 1


def save_label(user_id, name):
    with open(LABEL_FILE, "a") as f:
        f.write(f"{user_id},{name}\n")


def load_labels():
    labels = {}
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "r") as f:
            for line in f:
                uid, name = line.strip().split(",")
                labels[int(uid)] = name
    return labels

#  DATASET GENERATION 
def generate_dataset():
    name = simpledialog.askstring("Input", "Enter Name")
    if not name:
        return

    user_id = get_new_user_id()
    save_label(user_id, name)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img_id += 1
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            path = f"{DATASET_DIR}/user.{user_id}.{img_id}.jpg"
            cv2.imwrite(path, face)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
            cv2.imshow("Dataset Generation", frame)

        if cv2.waitKey(1) == 13 or img_id == 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Done", f"Dataset created for {name}")

#  TRAIN MODEL
def train_model():
    faces, ids = [], []

    for file in os.listdir(DATASET_DIR):
        img = Image.open(os.path.join(DATASET_DIR, file)).convert("L")
        img_np = np.array(img, "uint8")
        user_id = int(file.split(".")[1])

        faces.append(img_np)
        ids.append(user_id)

    if len(faces) == 0:
        messagebox.showerror("Error", "No dataset found!")
        return

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, np.array(ids))
    clf.write("classifier.xml")

    messagebox.showinfo("Success", "Training Completed!")

#  FACE RECOGNITION 
def detect_face():
    labels = load_labels()

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))

            id, pred = clf.predict(roi)
            confidence = int(100 * (1 - pred / 400))

            if confidence > CONFIDENCE_THRESHOLD:
                name = labels.get(id, "UNKNOWN")
            else:
                name = "UNKNOWN"

            cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 2)
            cv2.putText(img, f"{name} ({confidence}%)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,255), 2)

        cv2.imshow("Face Recognition", img)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()

#  BUTTONS 
tk.Button(window, text="Generate Dataset",
          font=("Arial", 14), bg="pink",
          command=generate_dataset).pack(pady=10)

tk.Button(window, text="Train Model",
          font=("Arial", 14), bg="orange",
          command=train_model).pack(pady=10)

tk.Button(window, text="Detect Face",
          font=("Arial", 14), bg="green", fg="white",
          command=detect_face).pack(pady=10)

window.mainloop()
