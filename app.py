"""
ROADEYE PRO — Multi-lane Vehicle Detection Using YOLO + Optical Flow
Developer: Vairam Venkatesh A | Vcodez Technologies Internship Project
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
# streamlit builds the web app UI — buttons, sliders, file uploads etc.
import streamlit as st

# cv2 is OpenCV — reads images/video, draws boxes, handles colors
import cv2

# numpy handles arrays of numbers (images are just big arrays of pixels)
import numpy as np

# YOLO is the object detection model from ultralytics library
from ultralytics import YOLO

# PIL opens image files that users upload
from PIL import Image

# tempfile creates temporary files to store uploaded content
import tempfile

# os helps us work with file paths
import os

# time is used for measuring processing speed
import time

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
# This MUST be the first streamlit command — sets browser tab title and layout
st.set_page_config(
    page_title="Roadeye Pro",
    page_icon="🚗",
    layout="wide"                # "wide" uses full screen width
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
# st.markdown with unsafe_allow_html=True lets us inject raw CSS into the page
st.markdown("""
<style>
    /* Dark professional background */
    .main { background-color: #0e1117; }
    
    /* Big gradient title */
    .title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #0077ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric cards — the count boxes */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem;
    }
    .metric-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00d4ff;
    }
    .metric-label {
        color: #aaa;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* Info box at the bottom */
    .info-box {
        background: #1a1f2e;
        border-left: 4px solid #0077ff;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        color: #ccc;
    }
</style>
""", unsafe_allow_html=True)

# ── TITLE ─────────────────────────────────────────────────────────────────────
# Display the big title using our custom CSS class
st.markdown('<div class="title">🚗 ROADEYE PRO</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Multi-Lane Vehicle Detection Using YOLO + Optical Flow | '
    'Vcodez Internship Project | Vairam Venkatesh A</div>',
    unsafe_allow_html=True
)

# Horizontal divider line
st.markdown("---")

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
# @st.cache_resource means: load this ONCE and reuse it every time
# Without this, the model would reload on every user interaction — very slow
@st.cache_resource
def load_model():
    """
    Downloads and loads YOLOv8 nano model.
    yolov8n.pt = nano version — smallest and fastest, good for deployment
    """
    # YOLO() automatically downloads the model weights if not already present
    model = YOLO("yolov8n.pt")
    return model

# Show a spinner while the model loads
with st.spinner("Loading YOLOv8 model..."):
    model = load_model()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# These are the class IDs for vehicles in the COCO dataset
# YOLO is trained on COCO which has 80 object types
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# Human-readable names for each class ID
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Colors for drawing boxes — BGR format (Blue, Green, Red) for OpenCV
# Each vehicle type gets a different color
COLORS = {
    2: (0, 255, 0),    # Car = Green
    3: (255, 0, 255),  # Motorcycle = Purple
    5: (0, 165, 255),  # Bus = Orange
    7: (0, 0, 255),    # Truck = Red
}

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# st.sidebar creates the left panel with controls
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Slider for confidence threshold
    # Higher = only show high-confidence detections (fewer but more accurate)
    # Lower = show more detections (more but some wrong ones)
    confidence = st.slider(
        "Detection Confidence",
        min_value=0.1,    # minimum value
        max_value=0.9,    # maximum value
        value=0.4,        # default value
        step=0.05,        # how much each step changes
        help="Higher = more accurate but may miss some vehicles"
    )
    
    # Checkbox to toggle optical flow overlay
    show_flow = st.checkbox(
        "Show Optical Flow",
        value=True,
        help="Visualizes vehicle movement direction and speed"
    )
    
    # Checkbox to show/hide lane dividers
    show_lanes = st.checkbox(
        "Show Lane Dividers",
        value=True,
        help="Divides frame into Left, Center, Right lanes"
    )
    
    # Number of lanes selector
    num_lanes = st.selectbox(
        "Number of Lanes",
        options=[2, 3, 4],
        index=1,           # default is index 1 = 3 lanes
        help="How many lanes to divide the road into"
    )
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **Roadeye Pro** detects and tracks vehicles across multiple road lanes in real time.
    
    **Technologies:**
    - YOLOv8 (Object Detection)
    - Optical Flow (Motion Tracking)
    - OpenCV (Image Processing)
    - Streamlit (Web App)
    
    **Developer:** Vairam Venkatesh A  
    **Internship:** Vcodez Technologies  
    **Duration:** Dec 2025 – Mar 2026
    """)

# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────

def get_lane_name(x_center, frame_width, num_lanes):
    """
    Given the x-position of a vehicle's center and the frame width,
    returns which lane the vehicle is in.
    
    Example: frame_width=900, num_lanes=3
    Lane 1 (Left):   x = 0 to 300
    Lane 2 (Center): x = 300 to 600
    Lane 3 (Right):  x = 600 to 900
    """
    # Calculate how wide each lane is
    lane_width = frame_width / num_lanes
    
    # Which lane number does this x position fall into?
    lane_num = int(x_center / lane_width)
    
    # Clamp to valid range (in case x_center equals frame_width exactly)
    lane_num = min(lane_num, num_lanes - 1)
    
    # Return lane name based on num_lanes
    if num_lanes == 2:
        return ["Left", "Right"][lane_num]
    elif num_lanes == 3:
        return ["Left", "Center", "Right"][lane_num]
    else:  # 4 lanes
        return ["Lane 1", "Lane 2", "Lane 3", "Lane 4"][lane_num]


def apply_optical_flow(frame1, frame2):
    """
    Calculates and visualizes optical flow between two consecutive frames.
    
    Optical Flow answers: "How did each pixel move between frame1 and frame2?"
    - Direction of movement → shown as color (hue)
    - Speed of movement → shown as brightness
    """
    # Convert both frames to grayscale
    # Optical flow works on single-channel (grayscale) images
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Farneback Dense Optical Flow
    # "Dense" means it calculates flow for EVERY pixel (vs sparse = only corners)
    # Parameters:
    # pyr_scale=0.5  → each pyramid level is half the size of previous
    # levels=3       → 3 levels in the image pyramid
    # winsize=15     → window size; larger = smoother flow
    # iterations=3   → number of iterations at each pyramid level
    # poly_n=5       → size of pixel neighborhood for polynomial expansion
    # poly_sigma=1.2 → standard deviation of Gaussian for smoothing
    # flags=0        → no special flags
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # flow shape is (height, width, 2)
    # flow[:,:,0] = horizontal movement (x direction)
    # flow[:,:,1] = vertical movement (y direction)
    
    # Convert x,y movement to polar form: magnitude (speed) and angle (direction)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image to visualize flow
    # HSV = Hue (color/direction), Saturation, Value (brightness/speed)
    hsv = np.zeros_like(frame1)
    
    # Saturation = 255 means fully saturated colors (vivid)
    hsv[..., 1] = 255
    
    # Hue channel encodes direction of movement
    # angle is in radians, we convert to degrees (0-180 for OpenCV hue range)
    hsv[..., 0] = angle * 180 / np.pi / 2
    
    # Value channel encodes speed — normalize to 0-255 range
    # Faster movement = brighter pixel
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert HSV visualization back to BGR for display
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flow_bgr


def process_frame(frame, prev_frame, confidence, show_flow, show_lanes, num_lanes):
    """
    The main processing function — runs on every single frame.
    
    Steps:
    1. Run YOLO detection
    2. Draw lane dividers
    3. Draw vehicle bounding boxes with labels
    4. Optionally overlay optical flow
    5. Count vehicles per lane
    6. Return processed frame and stats
    """
    height, width = frame.shape[:2]  # frame.shape = (height, width, channels)
    
    # ── STEP 1: RUN YOLO ─────────────────────────────────────────────────
    # model() runs inference — feeds the frame through the neural network
    # conf=confidence → only return detections above our threshold
    # verbose=False → don't print detection info to console
    results = model(frame, conf=confidence, verbose=False)
    result = results[0]  # results is a list, [0] gets the first (only) image
    
    # ── STEP 2: OPTICAL FLOW ─────────────────────────────────────────────
    # Only calculate flow if we have a previous frame to compare with
    if show_flow and prev_frame is not None:
        flow_viz = apply_optical_flow(prev_frame, frame)
        # cv2.addWeighted blends two images: 80% original + 20% flow
        # This makes flow visible without hiding the original video
        frame = cv2.addWeighted(frame, 0.8, flow_viz, 0.2, 0)
    
    # ── STEP 3: DRAW LANE DIVIDERS ────────────────────────────────────────
    lane_counts = {}
    if show_lanes:
        lane_width = width // num_lanes
        
        # Draw a vertical line for each lane boundary
        for i in range(1, num_lanes):
            x = i * lane_width
            # cv2.line(image, start_point, end_point, color, thickness)
            cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 2)
        
        # Initialize lane count dictionary
        if num_lanes == 2:
            lane_counts = {"Left": 0, "Right": 0}
        elif num_lanes == 3:
            lane_counts = {"Left": 0, "Center": 0, "Right": 0}
        else:
            lane_counts = {"Lane 1": 0, "Lane 2": 0, "Lane 3": 0, "Lane 4": 0}
    
    # ── STEP 4: DRAW DETECTIONS ───────────────────────────────────────────
    total_vehicles = 0
    vehicle_type_counts = {name: 0 for name in CLASS_NAMES.values()}
    
    # result.boxes contains all detected objects in this frame
    for box in result.boxes:
        # Get class ID (what type of object)
        class_id = int(box.cls[0])
        
        # Skip if not a vehicle
        if class_id not in VEHICLE_CLASSES:
            continue
        
        # Get confidence score (how sure YOLO is)
        conf_score = float(box.conf[0])
        
        # Get bounding box coordinates
        # xyxy format = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Get color for this vehicle type
        color = COLORS.get(class_id, (255, 255, 255))
        
        # Draw rectangle around the vehicle
        # thickness=2 means 2 pixels thick border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label text: "Car 87%" 
        label = f"{CLASS_NAMES[class_id]} {conf_score:.0%}"
        
        # Draw filled rectangle behind text for readability
        # First get the size of the text
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
        
        # Draw the label text
        # cv2.putText(image, text, position, font, scale, color, thickness)
        cv2.putText(
            frame, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
        
        # Count this vehicle
        total_vehicles += 1
        vehicle_type_counts[CLASS_NAMES[class_id]] += 1
        
        # Assign to lane if lane detection is on
        if show_lanes and lane_counts:
            center_x = (x1 + x2) / 2
            lane = get_lane_name(center_x, width, num_lanes)
            if lane in lane_counts:
                lane_counts[lane] += 1
    
    # ── STEP 5: DRAW LANE COUNT LABELS ON FRAME ───────────────────────────
    if show_lanes and lane_counts:
        lane_width = width // num_lanes
        for i, (lane_name, count) in enumerate(lane_counts.items()):
            # Position each label at the top center of its lane
            x_pos = i * lane_width + lane_width // 2 - 30
            label = f"{lane_name}: {count}"
            cv2.putText(
                frame, label, (x_pos, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
    
    # ── STEP 6: DRAW WATERMARK ────────────────────────────────────────────
    cv2.putText(
        frame, "ROADEYE PRO | Vairam Venkatesh",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    
    return frame, total_vehicles, vehicle_type_counts, lane_counts


# ── MAIN UI TABS ──────────────────────────────────────────────────────────────
# st.tabs creates clickable tab panels
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "ℹ️ How It Works"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE DETECTION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload a Traffic Image")
    st.markdown("Upload any road/traffic image and Roadeye Pro will detect and count all vehicles.")
    
    # File uploader widget — only accepts jpg, jpeg, png
    uploaded_image = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    if uploaded_image is not None:
        # Read the uploaded file as bytes and convert to numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        
        # cv2.imdecode converts bytes to an OpenCV image (numpy array)
        # cv2.IMREAD_COLOR means load as color (BGR) image
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Create two columns side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            # Convert BGR to RGB for display (streamlit uses RGB)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Process button
        if st.button("🔍 Detect Vehicles", type="primary", key="detect_img"):
            with st.spinner("Running YOLO detection..."):
                start_time = time.time()
                
                # Process the frame (no previous frame for image, so pass None)
                processed, total, type_counts, lane_counts = process_frame(
                    image.copy(), None, confidence, False, show_lanes, num_lanes
                )
                
                elapsed = time.time() - start_time
            
            with col2:
                st.markdown("**Detection Result**")
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Show metrics
            st.markdown("---")
            st.markdown("### 📊 Detection Results")
            
            # Create metric columns
            m_cols = st.columns(5)
            m_cols[0].metric("Total Vehicles", total)
            m_cols[1].metric("Cars", type_counts.get("Car", 0))
            m_cols[2].metric("Buses", type_counts.get("Bus", 0))
            m_cols[3].metric("Trucks", type_counts.get("Truck", 0))
            m_cols[4].metric("Processing Time", f"{elapsed:.2f}s")
            
            # Lane counts
            if lane_counts:
                st.markdown("### 🛣️ Vehicles Per Lane")
                lane_cols = st.columns(len(lane_counts))
                for i, (lane, count) in enumerate(lane_counts.items()):
                    lane_cols[i].metric(f"{lane} Lane", count)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO DETECTION
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Upload a Traffic Video")
    st.markdown("Upload an MP4 video. Roadeye Pro will process each frame and track vehicles across lanes.")
    
    uploaded_video = st.file_uploader(
        "Choose a video...",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    
    # Limit processing for demo purposes
    max_frames = st.slider(
        "Max frames to process",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="More frames = better demo but slower processing"
    )
    
    if uploaded_video is not None and st.button("🎬 Process Video", type="primary"):
        
        # Save uploaded video to a temp file
        # tempfile.NamedTemporaryFile creates a temporary file that auto-deletes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            tmp_path = tmp_file.name
        
        # Output path for processed video
        output_path = tmp_path.replace('.mp4', '_roadeye_output.mp4')
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()  # placeholder for updating text
        
        # Open the video
        cap = cv2.VideoCapture(tmp_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25  # default 25 if FPS not available
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_process = min(max_frames, total_frames)
        
        st.info(f"Video: {width}×{height} | {fps} FPS | Processing {frames_to_process} frames")
        
        # VideoWriter to save output
        # mp4v is the codec for MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Stats tracking
        all_counts = []
        prev_frame = None
        frame_idx = 0
        
        while cap.isOpened() and frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process this frame
            processed, total, type_counts, lane_counts = process_frame(
                frame, prev_frame, confidence, show_flow, show_lanes, num_lanes
            )
            
            # Write to output video
            out.write(processed)
            
            # Track stats
            all_counts.append(total)
            
            # Update previous frame
            prev_frame = frame.copy()
            frame_idx += 1
            
            # Update progress bar and status
            progress = frame_idx / frames_to_process
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx}/{frames_to_process} | Vehicles detected: {total}")
        
        # Close everything
        cap.release()
        out.release()
        
        status_text.text("✅ Processing complete!")
        progress_bar.progress(1.0)
        
        # Show summary stats
        st.markdown("### 📊 Processing Summary")
        s_cols = st.columns(3)
        s_cols[0].metric("Frames Processed", frame_idx)
        s_cols[1].metric("Max Vehicles in Frame", max(all_counts) if all_counts else 0)
        s_cols[2].metric("Avg Vehicles per Frame", f"{np.mean(all_counts):.1f}" if all_counts else "0")
        
        # Show a sample processed frame
        st.markdown("### 🖼️ Sample Output Frame")
        cap_check = cv2.VideoCapture(output_path)
        cap_check.set(cv2.CAP_PROP_POS_FRAMES, frame_idx // 2)  # middle frame
        ret, sample_frame = cap_check.read()
        cap_check.release()
        if ret:
            st.image(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        # Download button for processed video
        with open(output_path, 'rb') as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f,
                file_name="roadeye_pro_output.mp4",
                mime="video/mp4"
            )
        
        # Cleanup temp files
        os.unlink(tmp_path)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## How Roadeye Pro Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 YOLOv8 — Vehicle Detection
        
        **YOLO = You Only Look Once**
        
        Traditional object detectors scan an image multiple times. 
        YOLO is revolutionary because it scans the **entire image in one pass** 
        through a neural network.
        
        **How it works:**
        1. Divides the image into a grid
        2. Each grid cell predicts: "Is there a vehicle here?"
        3. If yes → draws a bounding box with confidence score
        4. Removes duplicate detections (NMS)
        5. Returns final detected vehicles
        
        **We detect:** Cars, Buses, Trucks, Motorcycles
        
        **Why YOLOv8?** It's the latest version — fastest and most accurate 
        for real-time applications.
        """)
    
    with col2:
        st.markdown("""
        ### 🌊 Optical Flow — Motion Tracking
        
        **Optical Flow tracks HOW objects move between frames**
        
        We use the **Farneback Dense Optical Flow** algorithm which 
        calculates movement for every single pixel.
        
        **How it works:**
        1. Takes two consecutive video frames
        2. Converts both to grayscale
        3. Calculates how each pixel moved (direction + speed)
        4. Converts movement to color visualization:
           - **Color (Hue)** = direction of movement
           - **Brightness** = speed of movement
        5. Overlays this on the original frame
        
        **Why useful?** Helps track which lane a vehicle is moving into,
        even between YOLO detection frames.
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🛣️ Multi-Lane Detection
    
    Roadeye Pro divides the road frame into equal vertical sections (2, 3, or 4 lanes).
    For each detected vehicle, it checks the **center x-coordinate** of the bounding box
    and determines which lane it falls into. This gives real-time per-lane vehicle counts.
    
    ### 👨‍💻 About This Project
    
    | Detail | Info |
    |--------|------|
    | Developer | Vairam Venkatesh A |
    | Registration | 212301488 |
    | College | Nazareth College of Arts & Science, Avadi |
    | Internship | Vcodez Technologies, Chennai |
    | Duration | December 2025 – March 2026 |
    | Role | Machine Learning Intern |
    """)
