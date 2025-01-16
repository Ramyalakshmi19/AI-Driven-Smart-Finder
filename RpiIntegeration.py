import os
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import nltk
from nltk import pos_tag, word_tokenize
import cv2
import torch
import matplotlib.pyplot as plt
from shapely.geometry import box as shapely_box

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # Load pre-trained YOLOv5 small model

# Define classes for small objects and potential surfaces
small_objects = ['remote', 'scissors', 'cup', 'book', 'magazine', 'figurine', 'lamp switch', 'potted plant', 'sofa', 'clock']
surface_objects = ['dining table', 'table', 'desk', 'shelf', 'chair', 'sofa']

# Get class indices for small objects and surfaces
small_object_indices = [model.names.index(obj) for obj in small_objects if obj in model.names]
surface_object_indices = [model.names.index(obj) for obj in surface_objects if obj in model.names]

# Function to check intersection between two bounding boxes
def check_intersection(bbox1, bbox2):
    box1 = shapely_box(*bbox1)  # Convert to shapely box
    box2 = shapely_box(*bbox2)  # Convert to shapely box
    return box1.intersects(box2)

# Function to calculate IoU (Intersection over Union)
def calculate_iou(bbox1, bbox2):
    box1 = shapely_box(*bbox1)  # Convert to shapely box
    box2 = shapely_box(*bbox2)  # Convert to shapely box
    intersection_area = box1.intersection(box2).area
    union_area = box1.union(box2).area
    return intersection_area / union_area if union_area != 0 else 0
# Update to ensure proper scaling of bounding boxes and visibility of text
def detect_objects_in_image(image_path):
    img = cv2.imread(image_path)  # Read the image using OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img_original = img.copy()  # Copy original image to keep it intact for visualization
    img = cv2.resize(img, (1280, 960))  # Resize the image to match the model input size if required

    # Perform prediction
    results = model(img)  # Forward pass

    # Store detected objects with their bounding boxes, labels, and confidence scores
    detected_objects = []
    for box in results.xywh[0]:
        class_index = int(box[5].item())
        label = model.names[class_index]
        bbox = box[:4].numpy()  # Extract bounding box as [x_center, y_center, width, height]
        confidence = box[4].item()  # Confidence score
        detected_objects.append({'label': label, 'bbox': bbox, 'confidence': confidence, 'class_index': class_index})

        # Draw bounding box and label on the image
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * img.shape[1])
        y1 = int((y_center - height / 2) * img.shape[0])
        x2 = int((x_center + width / 2) * img.shape[1])
        y2 = int((y_center + height / 2) * img.shape[0])

        # Draw the rectangle (bounding box) in red
        cv2.rectangle(img_original, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle
        # Draw label in white with black background
        cv2.putText(img_original, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_original, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)  # Shadow for better readability

    # Save the annotated image with bounding boxes
    annotated_image_path = "/home/fyp/Downloads/FYP_code/annotated_image.jpg"
    cv2.imwrite(annotated_image_path, cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR))

    return detected_objects, annotated_image_path



# Define a function to find and describe relationships between target and surrounding objects
def describe_object_location(target_label, detected_objects, default_surface="table"):
    target = next((obj for obj in detected_objects if obj['label'] == target_label), None)
    if not target:
        print(f"{target_label} not found.")
        return

    target_bbox = target['bbox']
    found_description = False

    for obj in detected_objects:
        if obj['label'] != target_label:
            # Check intersection with other objects
            iou = calculate_iou(target_bbox, obj['bbox'])
            print(f"IOU between {target_label} and {obj['label']}: {iou:.2f}")
            if check_intersection(target_bbox, obj['bbox']):
                # Determine the spatial relationship
                relationship = "on" if obj['label'] in surface_objects else "near"
                print(f"The {target_label} is {relationship} the {obj['label']}. (Confidence: {obj['confidence']:.2f})")
                found_description = True
                break

    # If no relevant surfaces or nearby objects were detected, use a default response
    if not found_description:
        print(f"The {target_label} was found near the {default_surface}.")

def extract_base_objects(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    objects = [word for word, tag in tagged if tag in ['NN', 'NNP', 'NNS', 'NNPS']]
    return objects

def get_external_microphone_device():
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0 and 'USB' in device['name']:
            print(f"Found external microphone: {device['name']} at device index {idx}")
            return idx
    return None

def listen_for_activation(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening continuously for 'Hey Buddy'...")

        while True:
            try:
                audio = recognizer.listen(source, timeout=None)
                text = recognizer.recognize_google(audio).lower()
                if "hey buddy" in text:
                    print("Activation phrase detected!")
                    return True
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

def record_audio_after_activation(duration=5, fs=44100):
    print("Recording command...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()
    return np.squeeze(audio)

def audio_to_text(audio, fs=44100):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(audio.tobytes(), fs, 2)
    try:
        text = recognizer.recognize_google(audio_data).lower()
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"API request error: {e}")
        return None

def capture_and_process_image():
    image_path = "/home/fyp/Downloads/FYP_code/Living_room.jpg"
    processed_images_dir = "/home/fyp/Downloads/FYP_code/processed_images"
    os.makedirs(processed_images_dir, exist_ok=True)

    # Use libcamera-still to capture an image
    print("Capturing image using libcamera-still...")
    result = os.system(f"libcamera-still -o {image_path}")
    if result != 0:
        print("Error: Failed to capture image using libcamera-still.")
        return

    print(f"Image captured and saved to {image_path}")

    # Load and process the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or invalid path.")
        return

    # Process images
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 100, 200)

    # Save processed images
    cv2.imwrite(f"{processed_images_dir}/gray_image.jpg", gray_image)
    cv2.imwrite(f"{processed_images_dir}/equalized_image.jpg", equalized_image)
    cv2.imwrite(f"{processed_images_dir}/blurred_image.jpg", blurred_image)
    cv2.imwrite(f"{processed_images_dir}/edges.jpg", edges)

    print(f"Processed images saved in {processed_images_dir}")

def main():
    recognizer = sr.Recognizer()

    # Get the device index for the external microphone
    device_index = get_external_microphone_device()

    if device_index is None:
        print("No external microphone found. Using default device.")
        microphone = sr.Microphone()
    else:
        microphone = sr.Microphone(device_index=device_index)

    # Get sample rate from the selected microphone
    sample_rate = int(sd.query_devices(device_index, 'input')['default_samplerate']) if device_index is not None else 44100

    # Ensure the device is available and we can adjust for ambient noise
    if listen_for_activation(recognizer, microphone):
        audio = record_audio_after_activation(duration=5, fs=sample_rate)
        text = audio_to_text(audio, fs=sample_rate)
        if text:
            print("Command:", text)
            objects = extract_base_objects(text)
            print(f"Sentence: '{text}' | Objects: {objects}")

            # Capture image and detect objects
            capture_and_process_image()
            detected_objects, annotated_image_path = detect_objects_in_image("/home/fyp/Downloads/FYP_code/Living_room.jpg")

            for obj in objects:
                describe_object_location(obj, detected_objects)


if __name__ == "__main__":
    main()
