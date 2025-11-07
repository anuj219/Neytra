import speech_recognition as sr
import pyttsx3
import keyboard

# ---------- TTS Helper ----------
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# ---------- Listen while key pressed ----------
def listen_while_pressed(key, prompt_text="Hold the key and speak..."):
    r = sr.Recognizer()
    speak(prompt_text)
    print(prompt_text)

    with sr.Microphone() as source:
        audio_data = None
        while True:
            if keyboard.is_pressed(key):     # when key pressed, start listening
                print("🎙 Listening...")
                audio_data = r.listen(source, phrase_time_limit=5)
            else:
                if audio_data:
                    break                    # stop once key is released
        try:
            text = r.recognize_google(audio_data).lower()
            print(f"✅ You said: {text}")
            speak(f"You said {text}")
            return textc
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            speak("Sorry, I didn’t catch that.")
            return None
        except Exception as e:
            print("⚠️ Error:", e)
            speak("An error occurred.")
            return None


# ---------- Main ----------
print("\n🎧 Voice Input Test – Press and hold 'V' key to talk\n")

while True:
    result = listen_while_pressed('v', "Press and hold V to speak.")
    if result:
        speak(f"You said {result}. Press C to confirm.")
        print("Press C to confirm, or any other key to retry...")

        key = keyboard.read_key()
        if key == 'c':
            speak("Confirmed.")
            print("✅ Confirmed input:", result)
            break
        else:
            speak("Retrying.")
            print("🔁 Retry.\n")

print("Test finished.")
speak("Test completed.")
