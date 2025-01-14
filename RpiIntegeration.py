import sounddevice as sd
import numpy as np
import speech_recognition as sr
import nltk
from nltk import pos_tag, word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_base_objects(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    objects = [word for word, tag in tagged if tag in ['NN', 'NNP', 'NNS', 'NNPS']]
    return objects

def get_external_microphone_device():
    devices = sd.query_devices()
    # Look for a device whose name contains "USB" (indicating it's likely your external mic)
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0 and 'USB' in device['name']:
            print(f"Found external microphone: {device['name']} at device index {idx}")
            return idx
    return None

def listen_for_activation(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)  # This requires 'source' to be inside a 'with' statement.
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

def record_audio_after_activation(duration=5, fs=44100):  # Set fs to the device's default rate (44100)
    print("Recording command...")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  
    return np.squeeze(audio)

def audio_to_text(audio, fs=44100):  # Ensure the same fs is used here
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

def load_and_process_image(image_path):
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or invalid path.")
        return
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 100, 200)
    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", gray_image)
    cv2.imshow("Equalized Image", equalized_image)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.imshow("Edges", edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function
def main():
    recognizer = sr.Recognizer()

    # Get the device index for the external microphone
    device_index = get_external_microphone_device()

    if device_index is None:
        print("No external microphone found. Using default device.")
        microphone = sr.Microphone()  # Use the system's default microphone if no external one is found
    else:
        microphone = sr.Microphone(device_index=device_index)  # Use the external microphone

    # Get sample rate from the selected microphone
    sample_rate = int(sd.query_devices(device_index, 'input')['default_samplerate']) if device_index is not None else 44100

    # Ensure the device is available and we can adjust for ambient noise
    if listen_for_activation(recognizer, microphone):
        audio = record_audio_after_activation(duration=5, fs=sample_rate)  # Adjust duration if needed
        text = audio_to_text(audio, fs=sample_rate)
        if text:
            print("Command:", text)
            objects = extract_base_objects(text)
            print(f"Sentence: '{text}' | Objects: {objects}")
            load_and_process_image("/home/ramyalakshmi/SmartSight/Living_Room.jpeg")

if __name__ == "__main__":
    main()

