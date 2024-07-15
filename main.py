import cv2
import streamlit as st
from gui_buttons import Buttons

# Initialize Buttons
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 100)
button.add_button("keyboard", 20, 180)
button.add_button("remote", 20, 260)
button.add_button("scissors", 20, 340)

colors = button.colors

# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)

# Initialize camera
cap = cv2.VideoCapture(0)  # Use the default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def process_frame():
    ret, frame = cap.read()
    if not ret:
        return None

    # Get active buttons list
    active_buttons = button.active_buttons_list()

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=0.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        color = colors[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

    # Display buttons
    button.display_buttons(frame)

    return frame

def main():
    st.title("Real-Time Object Detection with YOLOv4-Tiny")
    
    # Display buttons
    st.sidebar.header("Active Classes")
    for class_name, (x, y) in button.buttons.items():
        if st.sidebar.checkbox(class_name, key=class_name):
            button.activate_button(class_name)
        else:
            button.deactivate_button(class_name)

    frame_placeholder = st.empty()

    while True:
        frame = process_frame()
        if frame is None:
            break
        
        frame_placeholder.image(frame, channels="BGR")
    
if __name__ == '__main__':
    main()
