pip install ultralytics

import streamlit as st
from ultralytics import YOLO
import cv2
import os
import yaml

# Load fruit-price information from YAML file
with open('fruit_prices.yaml', 'r') as file:
    fruit_prices = yaml.safe_load(file)

# Initialize YOLO model
model = YOLO("../yolov8x.pt")
names = model.names

# Function to process image and generate receipt
def process_image(image_path):
    im0 = cv2.imread(image_path)
    h, w, _ = im0.shape

    # Predict objects using YOLO model
    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()

    # Initialize detected fruit counts
    detected_fruits = {names[int(cls)]: 0 for _, cls in zip(boxes, clss)}

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            # Count detected fruits
            detected_fruits[names[int(cls)]] += 1

    # Generate receipt
    receipt = ""
    total_price = 0
    for fruit_info in fruit_prices:
        fruit = fruit_info['fruit']
        if fruit in detected_fruits:
            price = fruit_info['price']
            count = detected_fruits[fruit]
            total_price += count * price
            receipt += f"{fruit.capitalize()}s: {count}*{price}: {count * price} Ksh\n"
    receipt += f"Total Price: {total_price} Ksh"
    return receipt

# Streamlit app UI
st.title("Fruit Detector and Receipt Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Process image and generate receipt
    receipt = process_image(uploaded_file.name)

    # Display receipt
    st.header("Receipt")
    st.text(receipt)
