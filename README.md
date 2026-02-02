# Face Recognition System using OpenCV & LBPH

### A real-time face recognition application built with Python, OpenCV, and Tkinter.

This project demonstrates a complete computer vision pipeline — from capturing face images using a webcam to training a machine learning model and recognizing people in real time with confidence scores.

---

### 👨‍💻 Author
**Harish Kumar**

---

### 🛠 Technologies Used
- Python  
- OpenCV  
- LBPH Face Recognizer  
- Haar Cascade Classifier  
- Tkinter (GUI)  
- NumPy  
- Pillow  

---

### 📖 Project Overview

The system works in three major stages:

1. **Dataset Generation**  
   The user enters their name and the webcam captures multiple face images.  
   These images are stored locally to create a training dataset.

2. **Model Training**  
   The captured images are processed and used to train an LBPH classifier.

3. **Face Recognition**  
   The trained model is used to detect and recognize faces from the webcam in real time, displaying the predicted name and confidence score.

---

### ⚙️ How It Works

- Haar Cascade detects faces from the camera feed  
- LBPH extracts features and compares them with trained faces  
- Tkinter provides a simple graphical interface  
- OpenCV handles image processing and camera access  

---

### ▶️ How to Run

Follow these steps carefully to run the Face Recognition System on your machine.

#### Step 1: Install Python
Make sure Python 3.8 or higher is installed.

Check version:
'''bash
python --version
'''
#### Step 2: Clone the Repository
git clone https://github.com/Harishyadav44/Face_Recognition_GUI.git
cd Face_Recognition_GUI

#### Step 3: Install Required Libraries
pip install -r requirements.txt

#### Step 4: Run the Application
python main.py

#### Step 5: Use the Application

###### 1. Click Generate Dataset
   Enter your name and look at the camera.
   The system will capture around 50 face images.

###### 2. Click Train Model
   This will train the LBPH face recognition model.

###### 3. Click Detect Face
   The webcam will open and display the recognized name with confidence.

#### Step 6: Exit
'''
    Press Enter on the keyboard to close the webcam window.
'''    
---

### 🖥 Application Buttons

Generate Dataset → Captures face images for a new user

Train Model → Trains the LBPH face recognizer

Detect Face → Starts real-time recognition using the webcam

---

### 🔒 Privacy Note

Dataset images and trained model files are not included in this repository for privacy reasons.
You can generate your own dataset using the “Generate Dataset” button.

---

### 🚀 Future Improvements

Attendance system integration

Face mask detection

Database connectivity

Web-based interface
