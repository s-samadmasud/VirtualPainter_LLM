import cv2
import numpy as np
import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from metaidigitcv.main import Handtracker
#import handtracking as ht
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit App Config
st.set_page_config(page_title="VirtualPainter AI", layout="wide")
st.title("🎨 VirtualPainter AI with LLM Integration")

# Sidebar controls
st.sidebar.header("🛠 Controls")
run = st.sidebar.checkbox("Run", value=True)
brush_thickness = st.sidebar.slider("Brush Thickness", 5, 50, 25)
eraser_thickness = st.sidebar.slider("Eraser Thickness", 50, 150, 100)

# Color Selection
colors = {
    "Purple": (255, 0, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Eraser": (0, 0, 0)
}
selected_color = st.sidebar.radio("Select Color", list(colors.keys()))
drawColor = colors[selected_color]

# Initialize Canvas in Session State (To Prevent Reset)
if "imgCanvas" not in st.session_state:
    st.session_state["imgCanvas"] = np.zeros((720, 1280, 3), np.uint8)

imgCanvas = st.session_state["imgCanvas"]  # Use the stored canvas

# Initialize Hand Detector
detector = Handtracker(detectionCon=0.85)

# Webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Streamlit columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    FRAME_WINDOW = st.image([])  # For webcam display

with col2:
    st.subheader("🎯 AI Prediction")
    output_text_area = st.empty()

def sendToLLM(image):
    """ Sends the drawn image to the LLM for prediction """
    pil_image = Image.fromarray(image)
    response = model.generate_content(["What does this drawing represent?", pil_image])
    return response.text

# Process Image with LLM on Button Click
if st.sidebar.button("🔍 Predict Drawing", key="predict_button"):
    prediction = sendToLLM(st.session_state["imgCanvas"])
    output_text_area.write(f"**LLM Prediction:** {prediction}")

# Clear Canvas Button
if st.sidebar.button("🗑 Clear Canvas", key="clear_canvas_button"):
    st.session_state["imgCanvas"] = np.zeros((720, 1280, 3), np.uint8)

# Streamlit App Loop
while run:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.identifyHands(img)
    lmList, bbox = detector.trackPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip
        fingers = detector.trackRaisedFingers(hands[0])

        # Selection Mode
        if fingers[1] and fingers[2]:  
            xp, yp = 0, 0

        # Drawing Mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            thickness = eraser_thickness if drawColor == (0, 0, 0) else brush_thickness
            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
            xp, yp = x1, y1

    # Convert Canvas for Display
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Update Session State (to keep the drawn canvas persistent)
    st.session_state["imgCanvas"] = imgCanvas  

    # Display Streamlit Webcam Feed
    FRAME_WINDOW.image(img, channels="BGR")

    # Stop the App
    if not run:
        break

cap.release()
cv2.destroyAllWindows()
