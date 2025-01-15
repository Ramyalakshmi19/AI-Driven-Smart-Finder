import os
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import nltk
from nltk import pos_tag, word_tokenize
import cv2

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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
            capture_and_process_image()

if __name__ == "__main__":
    main()
