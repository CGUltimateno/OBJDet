import cv2
import os
import tkinter as tk
from tkinter import filedialog
from threading import Thread

import numpy as np
from gtts import gTTS
from playsound import playsound

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")


def initialize_video_capture():
    return cv2.VideoCapture(0)


def generate_speech(label):
    text_to_speech = f"I see a {label}"
    tts = gTTS(text=text_to_speech, lang='en')
    tts.save(os.path.join(output_directory, f"{label}.mp3"))
    playsound(os.path.join(output_directory, f"{label}.mp3"))


def detect_objects(frame):
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(output_layers_names)

    conf_threshold = 0.5
    nms_threshold = 0.4

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detected_labels = [classes[class_ids[i]] for i in indices]
    detected_boxes = [boxes[i] for i in indices]

    return detected_labels, detected_boxes


def display_detection_result(image, detected_labels, detected_boxes):
    for i, box in enumerate(detected_boxes):
        x, y, w, h = box
        color = (0, 255, 0)  # Green color
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, detected_labels[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Object Detection", image)
    cv2.waitKey(1)


def object_detection_from_images():
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
    if file_path:
        image = cv2.imread(file_path)
        detected_labels, detected_boxes = detect_objects(image)
        display_detection_result(image, detected_labels, detected_boxes)


def live_camera_detection():
    video = initialize_video_capture()
    labels = []

    while True:
        ret, frame = video.read()
        detected_labels, detected_boxes = detect_objects(frame)

        for label in detected_labels:
            if label not in labels:
                labels.append(label)
                generate_speech(label)

        display_detection_result(frame, detected_labels, detected_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.title("Object Detection")

    button_image_detection = tk.Button(root, text="Object Detection from Images", command=object_detection_from_images)
    button_image_detection.pack()

    button_live_detection = tk.Button(root, text="Live Camera Detection",
                                      command=lambda: Thread(target=live_camera_detection).start())
    button_live_detection.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
