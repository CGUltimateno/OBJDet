import cv2
import os
from gtts import gTTS
from playsound import playsound

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

import cvlib as cv
from cvlib.object_detection import draw_bbox


def initialize_video_capture():
    return cv2.VideoCapture(0)


def generate_speech(label):
    text_to_speech = f"I see a {label}"
    tts = gTTS(text=text_to_speech, lang='en')
    tts.save(os.path.join(output_directory, f"{label}.mp3"))
    playsound(os.path.join(output_directory, f"{label}.mp3"))


def detect_objects(frame):
    return cv.detect_common_objects(frame)


def main():
    video = initialize_video_capture()
    labels = []

    # Variable to limit speech generation
    speech_limit = 5
    speech_count = 0

    while True:
        ret, frame = video.read()
        bbox, label, conf = detect_objects(frame)
        output_image = draw_bbox(frame, bbox, label, conf)

        cv2.imshow("Object Detection", output_image)

        for item in label:
            if item not in labels:
                labels.append(item)
                generate_speech(item)
                speech_count += 1

        key = cv2.waitKey(1)

        if key == ord("q") or speech_count >= speech_limit:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
