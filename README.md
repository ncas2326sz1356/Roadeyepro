# 🚗 ROADEYE PRO
### Multi-Lane Vehicle Detection Using YOLO + Optical Flow
**Developer:** Vairam Venkatesh A | **Internship:** Vcodez Technologies | **Dec 2025 – Mar 2026**

---

## 📁 PROJECT STRUCTURE
```
roadeye_pro/
├── app.py              ← Main Streamlit application (ALL the code is here)
├── requirements.txt    ← Python packages needed
├── packages.txt        ← Linux system packages needed for deployment
└── README.md           ← This file
```

---

## 🖥️ HOW TO RUN LOCALLY (on your Linux machine)

Open terminal and run these commands ONE BY ONE:

```bash
# Step 1: Go into the project folder
cd roadeye_pro

# Step 2: Install all required packages
pip install -r requirements.txt

# Step 3: Run the app
streamlit run app.py
```

Your browser will automatically open at: **http://localhost:8501**

---

## 🚀 HOW TO DEPLOY FREE ON STREAMLIT CLOUD

### Step 1 — Create GitHub account
Go to **github.com** → Sign up (free)

### Step 2 — Create a new repository
- Click the **+** button → "New repository"
- Name it: `roadeye-pro`
- Set to **Public**
- Click "Create repository"

### Step 3 — Upload your files
- Click "uploading an existing file"
- Drag and drop these 3 files:
  - `app.py`
  - `requirements.txt`
  - `packages.txt`
- Click "Commit changes"

### Step 4 — Deploy on Streamlit Cloud
- Go to **share.streamlit.io**
- Sign in with your GitHub account
- Click "New app"
- Select your repository: `roadeye-pro`
- Main file path: `app.py`
- Click **"Deploy!"**

### Step 5 — Wait 3-5 minutes
Streamlit Cloud will install everything automatically.
Your live URL will be something like:
**https://vairam-roadeye-pro.streamlit.app**

**Share this URL in your interview!** ✅

---

## 🎤 INTERVIEW PREPARATION — Read this carefully tonight!

---

### Q1: "Tell me about your project"
**Answer:**
> "Roadeye Pro is a real-time multi-lane vehicle detection and tracking system I built during my Machine Learning internship at Vcodez Technologies, Chennai. The system takes traffic video as input, detects vehicles like cars, buses, and trucks using YOLOv8, tracks their movement using Optical Flow, and counts how many vehicles are in each road lane at any given moment. I built it as a Streamlit web application so anyone can use it through a browser without any coding knowledge."

---

### Q2: "What is YOLO and why did you use it?"
**Answer:**
> "YOLO stands for You Only Look Once. It's a deep learning object detection algorithm that processes the entire image in a single forward pass through a neural network, unlike older methods that scanned the image multiple times. I chose YOLOv8 — the latest version — because it's extremely fast, accurate, and perfect for real-time video processing. It's pretrained on the COCO dataset which includes 80 object classes including cars, buses, trucks and motorcycles — exactly what I needed."

---

### Q3: "What is Optical Flow?"
**Answer:**
> "Optical Flow is a computer vision technique that measures how pixels move between two consecutive video frames. I used the Farneback Dense Optical Flow algorithm, which calculates movement for every single pixel in the frame. The direction of movement is encoded as color and the speed is encoded as brightness. In Roadeye Pro, I blend this optical flow visualization with the original video frame to show not just where vehicles are, but how fast and in which direction they're moving."

---

### Q4: "How does lane detection work in your project?"
**Answer:**
> "The lane detection works by dividing the video frame into equal vertical sections — 2, 3, or 4 lanes depending on the user's selection. For each vehicle that YOLO detects, I calculate the center x-coordinate of its bounding box and check which vertical section it falls into. That determines which lane the vehicle is in. I then display real-time per-lane counts on the video frame itself."

---

### Q5: "What libraries did you use?"
**Answer:**
> "I used five main libraries:
> - **Ultralytics** for the YOLOv8 model
> - **OpenCV** for reading video frames, drawing bounding boxes, and image processing
> - **NumPy** for numerical operations on image arrays
> - **Streamlit** to build the web application interface
> - **Pillow** for handling image file uploads"

---

### Q6: "What was your biggest challenge?"
**Answer:**
> "The biggest challenge was performance — processing every frame of a video in real time is computationally heavy. I solved this by using YOLOv8 Nano, which is the smallest and fastest version of the model. I also added a configurable confidence threshold so users can trade off accuracy vs speed. And I used OpenCV's headless version for deployment since servers don't have a display screen."

---

### Q7: "What is confidence threshold?"
**Answer:**
> "Confidence threshold is a value between 0 and 1 that filters YOLO's detections. When YOLO detects an object, it also gives a confidence score — how sure it is. If I set threshold to 0.4, it means I only accept detections where YOLO is at least 40% confident. A higher threshold gives fewer but more accurate detections. A lower threshold shows more detections but some may be wrong."

---

### Q8: "Why Streamlit for the frontend?"
**Answer:**
> "Streamlit is a Python library that lets you build web applications entirely in Python — no HTML, CSS, or JavaScript needed. Since I'm a Python developer, this was the fastest way to build a professional-looking web UI. It handles file uploads, buttons, sliders, progress bars, and image display all with simple Python commands. It also integrates perfectly with machine learning libraries."

---

### Q9: "Where is this deployed?"
**Answer:**
> "It's deployed on Streamlit Cloud, which offers free hosting for Streamlit applications. You just connect your GitHub repository and it automatically installs all dependencies and serves the application. The live URL is [your URL here]."

---

### Q10: "What would you improve if given more time?"
**Answer:**
> "Three things: First, I would add vehicle speed estimation using the optical flow magnitude and camera calibration data. Second, I would implement vehicle counting across a virtual trip wire — a line drawn across the road — to count total vehicles passing a point over time. Third, I would add a live webcam feed option so it works on real traffic cameras, not just uploaded videos."

---

## 📝 KEY CONCEPTS TO REMEMBER

| Term | Simple Explanation |
|------|-------------------|
| YOLO | Detects WHAT is in the image and WHERE |
| Optical Flow | Detects HOW things are MOVING |
| Bounding Box | Rectangle drawn around detected object |
| Confidence Score | How sure YOLO is (0 to 1) |
| COCO Dataset | 80-class dataset YOLO was trained on |
| BGR vs RGB | OpenCV uses BGR order, most others use RGB |
| Dense Optical Flow | Calculates movement for every pixel |
| Farneback | The specific algorithm used for optical flow |
| YOLOv8n | Nano = smallest, fastest YOLO version |
| Streamlit | Python library to build web apps |

---

*"I built this during my 3-month Machine Learning internship at Vcodez Technologies, Chennai. 
Rated Excellent by HR management." — Vairam Venkatesh A*
