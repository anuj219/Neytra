# To do - 
# multiple encodings per person
# If unknown person detected, y to add new ... else a to append to existing name
    # max 3 encodings allowed per person
    # thus, if we add it to existing name, that name's oldest (first) encoding gets deleted and new one gets added

from time import time
import cv2
import face_recognition
import pickle
import os
import pyttsx3

# Initialize TTS
import threading     # cause running it normally was not working
# reason - 
# runAndWait was a blocking function, as well as VideoCapture.
# so when either of them was running, it used to occupy the thread.

# Before
# Main Thread:
#   |--> run OpenCV loop
#   |--> call pyttsx3 (blocks here)

# After
# Main Thread:             Thread 2 (Speech)
#   |--> run OpenCV loop     |--> init pyttsx3
#   |--> detect faces        |--> say(text)
#   |--> call speak()        |--> runAndWait()
#                            |--> finish & die


# ---------- Helper functions ----------
def speak(text):
    threading.Thread(target=_speak_thread, args=(text,), daemon=True).start()
    # thus we created a seperate small thread for speak

def _speak_thread(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


DB_PATH = 'faces.pkl'


def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'rb') as f:
            data = pickle.load(f)
            return data['encodings'], data['names']
    else:
        return [], []

def save_database(encodings, names):
    with open(DB_PATH, 'wb') as f:
        pickle.dump({'encodings': encodings, 'names': names}, f)

# ---------- Main Program ----------

encodings, names = load_database()
speak(f"Loaded {len(names)} known faces.")
print(f"Loaded {len(names)} known faces.")
print(encodings)
print(names)

cap = cv2.VideoCapture(0)
speak("Neytra activated.")

recognize = False  # initially False

last_seen = None
last_seen_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        speak("Neytra shutting down.")
        break
    elif key == ord('r'):
        recognize = not recognize  # toggle recognition

    if recognize:
        # face recognition logic
        recognize = False  # reset until next key press
        small = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)


        if not face_locations:
            speak("No face detected.")
            print("No face detected.\n")
            recognize = False   
            continue


        for face_encoding, loc in zip(face_encodings, face_locations):
            if encodings:
                matches = face_recognition.compare_faces(encodings, face_encoding)
                face_distances = face_recognition.face_distance(encodings, face_encoding)
                best_match = min(range(len(face_distances)), key=face_distances.__getitem__)
                
                if matches[best_match] and face_distances[best_match] < 0.45:    # can tweak this value here, to include different face angles as well
                    name = names[best_match]
                    speak(f"{name} detected.")
                    print(f"{name} detected. \n")
                    y1, x2, y2, x1 = [v*4 for v in loc]
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    # after recognizing:
                    last_seen = (name, (x1, y1, x2, y2))
                    last_seen_time = time()
                else:
                    speak("Unknown person. \n")
                    y1, x2, y2, x1 = [v*4 for v in loc]
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(frame, "Unknown", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                    # Add new face
                    choice = input("Add this face? (y/n): ").strip().lower()
                    if choice == 'y':
                        name = input("Enter name: ").strip()
                        if name in names:
                            speak(f"{name} already exists in database.")
                            print(f"{name} already exists. \n")
                        else:
                            encodings.append(face_encoding)
                            names.append(name)
                            save_database(encodings, names)
                            speak(f"{name} added to memory.")
            else:
                speak("Database empty. Adding first face.")
                name = input("Enter First name: ").strip()
                encodings.append(face_encoding)
                names.append(name)
                save_database(encodings, names)
                speak(f"{name} saved as first known person.")

    # Add this just before cv2.imshow
    if last_seen and (time() - last_seen_time < 3):  # show for 3 seconds
        name, (x1, y1, x2, y2) = last_seen
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Neytra View', frame)


cap.release()
cv2.destroyAllWindows()
speak("Neytra shutting down.")
