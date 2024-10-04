import sounddevice as sd
import numpy as np
import speech_recognition as sr
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_base_objects(sentence):
    doc = nlp(sentence)
    objects = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and token.dep_ not in ['det', 'poss']:
            objects.append(token.text)
    
    return list(objects)


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


def record_audio_after_activation(duration=5, fs=16000):
    print("Recording command...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  
    return np.squeeze(audio) 


def audio_to_text(audio, fs=16000):
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
    microphone = sr.Microphone()

    if listen_for_activation(recognizer, microphone):
        audio = record_audio_after_activation(duration=5)  
        text = audio_to_text(audio)
        if text:
            print("Command:", text)
            objects = extract_base_objects(text)
            print(f"Sentence: '{text}' | Objects: {objects}")
            load_and_process_image("/home/ramyalakshmi/SmartSight/Living_room.png")


if __name__ == "__main__":
    main()
