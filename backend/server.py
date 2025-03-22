from flask import Flask
from flask import Flask, jsonify
import psutil
import os
import webbrowser
import mediapipe as mp
import pyautogui
import threading
import cv2
import time
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import cvzone
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/partitions', methods=['GET'])
def open_parti():
    partitions = psutil.disk_partitions(all=True)
    folders = [partition.device for partition in partitions]
    return jsonify({'folders': folders})


def get_directories_in_drive(f_name):
    try:
        directories = [name for name in os.listdir(f_name) if os.path.isdir(os.path.join(f_name, name))]
        return jsonify({'directories': directories})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/partition/<path:f_name>', methods=['GET'])
def open_parti_spec(f_name):
    partitions = psutil.disk_partitions(all=True)
    folders = [partition.device for partition in partitions]
    directory_list = []
    print(f_name)
    if f_name in folders:
        directory_list = get_directories_in_drive(f_name).json.get('directories', [])

    return jsonify({'directories': directory_list, 'root': f_name})


@app.route('/partition/<path:f_name0>/<path:f_name1>', methods=['GET'])
def open_parti_spec2(f_name0, f_name1):
    print(f_name0, f_name1)
    f_name0 = f_name0.replace('/', '\\')
    f_name1 = f_name1.replace('/', '\\') if f_name1 else ''
    path = os.path.join(f_name0, f_name1)
    is_file = os.path.isfile(path)

    if is_file:
        webbrowser.open(path)

    directory_list = []
    file_list = []
    if os.path.exists(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            file_list.append(item)

    root = os.path.join(f_name0, f_name1)

    return jsonify({'files': file_list, 'root': root, 'parent1': f_name1})


# Global thread control
virtual_mouse_thread = None
stop_virtual_mouse = threading.Event()


def run_virtual_mouse():
    """Function to start the virtual mouse with hand tracking"""
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()

    while not stop_virtual_mouse.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

                index_x, index_y = 0, 0
                thumb_x, thumb_y = 0, 0
                middle_x, middle_y = 0, 0

                for id, landmark in enumerate(hand.landmark):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    if id == 8:  # Index finger
                        index_x = x
                        index_y = y

                    if id == 4:  # Thumb
                        thumb_x = x
                        thumb_y = y

                    if id == 12:  # Middle finger
                        middle_x = x
                        middle_y = y

                # Draw circles on fingers
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)
                cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255), -1)
                cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 255), -1)

                # Calculate distances
                distance_index_middle = ((middle_x - index_x) ** 2 + (middle_y - index_y) ** 2) ** 0.5
                distance_index_thumb = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

                if distance_index_middle < 50:
                    pyautogui.click()
                    pyautogui.sleep(1)
                elif distance_index_thumb < 80:
                    pyautogui.moveTo(index_x * (screen_width / frame_width), index_y * (screen_height / frame_height))

        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@app.route('/activate-vm', methods=['POST'])
def activate_vm():
    """Flask endpoint to start the virtual mouse in a separate thread"""
    global virtual_mouse_thread, stop_virtual_mouse

    if virtual_mouse_thread and virtual_mouse_thread.is_alive():
        return jsonify({"status": "Virtual mouse is already running"}), 400

    stop_virtual_mouse.clear()
    virtual_mouse_thread = threading.Thread(target=run_virtual_mouse)
    virtual_mouse_thread.start()
    return jsonify({"status": "Virtual mouse activated"})


@app.route('/stop-vm', methods=['POST'])
def stop_vm():
    """Flask endpoint to stop the virtual mouse"""
    global stop_virtual_mouse, virtual_mouse_thread

    if virtual_mouse_thread and virtual_mouse_thread.is_alive():
        stop_virtual_mouse.set()
        virtual_mouse_thread.join()
        return jsonify({"status": "Virtual mouse stopped"})

    return jsonify({"status": "No virtual mouse running"}), 400


@app.route('/gesture-scroll/start', methods=['POST'])
def execute_gesture_scroll():
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    # OpenCV capture
    cap = cv2.VideoCapture(0)

    def detect_scroll_gesture(landmarks):
        # Check the position of the thumb and index fingers
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Thumb Down Gesture: Thumb tip is below the base of the thumb (landmark 2)
        thumb_base = landmarks[mp_hands.HandLandmark.THUMB_CMC]

        # Index Finger Up Gesture: Index tip is extended (above other fingers)
        if thumb_tip.y > thumb_base.y and index_tip.y < thumb_tip.y:
            # Thumb pointing down, trigger scroll down
            return "scroll_down"
        elif index_tip.y < thumb_tip.y:
            # Index finger extended, trigger scroll up
            return "scroll_up"

        return None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the gesture based on hand landmarks
                gesture = detect_scroll_gesture(landmarks.landmark)
                if gesture == "scroll_down":
                    pyautogui.scroll(-10)  # Scroll down
                    cv2.putText(frame, "Thumb Down: Scrolling Down", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)
                elif gesture == "scroll_up":
                    pyautogui.scroll(10)  # Scroll up
                    cv2.putText(frame, "Index Finger: Scrolling Up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Hand Gestures for Scrolling", frame)

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "Virtual mouse activated"})


@app.route('/draganddrop/start', methods=['POST'])
def execDragAndDrop():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Screen dimensions
    screen_width, screen_height = pyautogui.size()

    # Smoothing parameters
    smooth_factor = 7  # Higher value = smoother, but less responsive
    prev_x, prev_y = 0, 0

    # Gesture stability threshold
    gesture_cooldown = 0.5  # Minimum time (in seconds) between gestures
    last_gesture_time = time.time()

    def smooth_movement(curr_x, curr_y, prev_x, prev_y, factor):
        """
        Smooth cursor movement using weighted average.
        """
        smooth_x = (curr_x + prev_x * (factor - 1)) / factor
        smooth_y = (curr_y + prev_y * (factor - 1)) / factor
        return int(smooth_x), int(smooth_y)

    def fingers_up(hand_landmarks):
        """
        Determine which fingers are up using hand landmarks.
        """
        fingers = []
        tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips

        # Check thumb
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)

        # Check other fingers
        for tip in tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    # Webcam loop
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect fingers up
                finger_states = fingers_up(hand_landmarks)

                # Get index finger tip position
                index_tip = hand_landmarks.landmark[8]
                index_x = int(index_tip.x * screen_width)
                index_y = int(index_tip.y * screen_height)

                # Smooth cursor movement
                index_x, index_y = smooth_movement(index_x, index_y, prev_x, prev_y, smooth_factor)
                prev_x, prev_y = index_x, index_y

                # Move cursor when only the index finger is up
                if finger_states == [0, 1, 0, 0, 0]:
                    pyautogui.moveTo(index_x, index_y)
                    cv2.putText(frame, "Moving Cursor", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Click when both index and middle fingers are up
                elif finger_states == [0, 1, 1, 0, 0]:
                    if time.time() - last_gesture_time > gesture_cooldown:
                        pyautogui.click()
                        last_gesture_time = time.time()
                    cv2.putText(frame, "Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Drag when index, middle, and ring fingers are up
                elif finger_states == [0, 1, 1, 1, 0]:
                    pyautogui.mouseDown()
                    pyautogui.moveTo(index_x, index_y)
                    cv2.putText(frame, "Dragging", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Drop when all fingers are up
                elif finger_states == [0, 1, 1, 1, 1]:
                    pyautogui.mouseUp()
                    cv2.putText(frame, "Drop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show frame
        cv2.imshow("Virtual Mouse", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"status": "Virtual mouse activated"})


virtual_rmouse_thread = None


def run_virtual_rmouse():
    cam = cv2.VideoCapture(0)
    print(cv2.__version__)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_w, screen_h = pyautogui.size()
    while True:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0))
                if id == 1:
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
            if (left[0].y - left[1].y) < 0.0085:
                pyautogui.click()
                pyautogui.sleep(1)
        cv2.imshow('Eye Controlled Mouse', frame)
        key = cv2.waitKey(1)
        if key is ord('q'):
            global virtual_rmouse_thread
            if not virtual_rmouse_thread or not virtual_rmouse_thread.is_alive():
                virtual_rmouse_thread = threading.Thread(target=run_virtual_rmouse)
                virtual_rmouse_thread.start()
            break
    cv2.destroyAllWindows()  # Close OpenCV windows
    return jsonify({"status": "Retina mouse activated"})


@app.route('/retinamouse/start', methods=['POST'])
def activateRM():
    global virtual_rmouse_thread
    if not virtual_rmouse_thread or not virtual_rmouse_thread.is_alive():
        virtual_rmouse_thread = threading.Thread(target=run_virtual_rmouse)
        virtual_rmouse_thread.start()
        return jsonify({"status": "Virtual mouse activated"})


hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()


@app.route('/presentationcontroller/start', methods=['POST'])
def execPresentation(Fname='Pitch'):
    # Variables
    width, height = 1280, 720
    folderPath = 'Presentation/' + Fname

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Get list of images
    pathImages = sorted(os.listdir(folderPath), key=len)
    print(pathImages)

    # Variables
    imgNumber = 0
    hs, ws = int(120 * 1), 213
    gestureThreshold = 300
    buttonPressed = False
    buttonCounter = 0
    buttonDelay = 30
    annotations = [[]]
    annotationNumber = -1
    annotationStart = False

    # Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        # Import images
        success, img = cap.read()
        img = cv2.flip(img, 1)
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv2.imread(pathFullImage)

        # Resize the image to fit the screen
        imgCurrent = cv2.resize(imgCurrent, (width + 250, height + 70))

        hands, img = detector.findHands(img)

        if hands and buttonPressed is False:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            print(fingers)
            lmList = hand['lmList']

            # Constrain value for easier drawing
            indexFinger = lmList[8][0], lmList[8][1]
            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))

            indexFinger = xVal, yVal
            # Gesture 1
            # Thumb = Backward
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                if imgNumber > 0:
                    imgNumber -= 1
                    buttonPressed = True

            # Gesture 2
            # Pinki Finger = Forward
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    buttonPressed = True

            # Gesture 3
            # Show Pointer = Second and Index Finger
            if fingers == [0, 1, 1, 0, 0]:
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)

            # Gesture 4
            # DrawPointer
            if fingers == [0, 1, 0, 0, 0]:
                if annotationStart is False:
                    annotationStart = True
                    annotationNumber += 1
                    annotations.append([])
                cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
                annotations[annotationNumber].append(indexFinger)
            else:
                annotationStart = False

            # Gesture 5
            # Erase
            if fingers == [0, 1, 1, 1, 0]:
                if annotations:
                    annotations.pop(-1)
                    annotationNumber -= 1
                    buttonPressed = True

        # Button Pressed Iterations
        if buttonPressed:
            buttonCounter += 1
            if buttonCounter > buttonDelay:
                buttonCounter = 0
                buttonPressed = False

        for i in range(len(annotations)):
            for j in range(len(annotations[i])):
                if j != 0:
                    cv2.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 0, 200), 12)

            # Adding webcam image in slide
            imgSmall = cv2.resize(img, (ws, hs))
            h, w, _ = imgCurrent.shape
            imgCurrent[0:hs, w - ws:w] = imgSmall

        cv2.imshow("Image", img)
        cv2.imshow("Slides", imgCurrent)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    return jsonify({"status": "Success in Presentation Controller"})


@app.route('/signlanguage/start', methods=['POST'])
def speciallyabled():
    # Variables
    width, height = 500, 500

    # Camera setup
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Hand Detector
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    while True:
        # Import images
        success, img = cap.read()
        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img)

        if hands and len(hands) == 1:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            print(fingers)
            lmList = hand['lmList']

            # Thumb tip and index tip coordinates
            thumbTip = lmList[4][0], lmList[4][1]
            indexTip = lmList[8][0], lmList[8][1]

            # Calculate distance between thumb tip and index tip
            distance = np.linalg.norm(np.array(thumbTip) - np.array(indexTip))

            # Constrain value for easier drawing
            indexFinger = lmList[8][0], lmList[8][1]
            xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
            yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height]))

            indexFinger = xVal, yVal

            # Gesture1
            # Display "Nice" if the distance is very less
            if distance < 30:  # Adjust the threshold value as needed
                text = 'Fine'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            if fingers == [1, 1, 1, 1, 1]:
                text = 'Wait'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Thumb = Drink Water
            elif fingers == [1, 0, 0, 0, 0]:
                text = 'Drink Water'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture2
            # Thumb + Index = Smile
            elif fingers == [1, 1, 0, 0, 0]:
                text = 'Smile'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture 3
            # Thumb + Index + Middle Finger = Understood
            elif fingers == [1, 1, 1, 0, 0]:
                text = 'Understood'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture 4
            # Thumb + Index + Middle + Ring = Rock On
            elif fingers == [0, 1, 0, 0, 1]:
                text = 'Rock On'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture 5
            # Index = No
            elif fingers == [0, 1, 0, 0, 0]:
                text = 'No'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture 6
            # Pinky and Thumb = Call
            elif fingers == [1, 0, 0, 0, 1]:
                text = 'Call'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture 7
            # Thumb + Index + Pinky = I Love U
            elif fingers == [1, 1, 0, 0, 1]:
                text = 'I Love U'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture 8
            # Index + Middle + Ring = Slow
            elif fingers == [0, 1, 1, 1, 0]:
                text = 'Slow'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture 9
            # Index + Middle = Louder
            elif fingers == [0, 1, 1, 0, 0]:
                text = 'Louder'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Gesture 10
            # Pinky = Repeat
            elif fingers == [0, 0, 0, 0, 1]:
                text = 'Repeat'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            elif fingers == [0, 0, 1, 0, 0]:
                text = 'Go To Hell'
                cv2.putText(img, text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()  # Close OpenCV windows
    return jsonify({"status": "Successfully Run Sign Language"})


@app.route('/interactivereading/start', methods=['POST'])
def execInteractiveReading():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)

    textList = ["Welcome To", "my Python Project . ",
                "It is an", "interactive reading Panel .",
                "If hope you like it . "]

    while True:
        success, img = cap.read()
        imgText = np.zeros_like(img)
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]

            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # Finding the distance or depth
            f = 500
            d = (W * f) / w
            print(d)

            cvzone.putTextRect(img, f'Depth : {int(d)} cm', (face[10][0] - 100, face[10][1] - 50), scale=2)

            for i, text in enumerate(textList):
                singleHeight = 20 + int(d / 5)
                scale = 0.4 + (int(d / 10) * 10) / 80
                cv2.putText(imgText, text, (50, 50 + (i * singleHeight)), cv2.FONT_ITALIC, scale, (255, 255, 255), 2)

        imgStacked = cvzone.stackImages([img, imgText], 2, 1)
        cv2.imshow("Image", imgStacked)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    return jsonify({'status': 'Success'})


import comtypes
import pyttsx3
import speech_recognition as sr
import PyPDF2
import psutil
import datetime


def voice(flag=1):
    # Initialize COM explicitly
    comtypes.CoInitialize()

    # Initialize pyttsx3 engine
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voices', voices[0].id)

    def speak(audio):
        engine.say(audio)
        engine.runAndWait()

    # to convert voice into text  Take command
    def takecommand():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print('Listening...')
            r.pause_threshold = 1
            r.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                speak('Timeout occurred. Please try again.')
                return 'none'

        try:
            print('Recognizing...')
            query = r.recognize_google(audio, language='en-in')
            print(f'User said: {query}')
        except Exception as e:
            speak('Sorry, I could not understand that. Please try again.')
            return 'none'
        return query

    # To wish
    def wish():
        hour = int(datetime.datetime.now().hour)
        if hour >= 0 and hour <= 12:
            speak('Good morning')
        elif hour >= 12 and hour <= 18:
            speak('Good Afternoon')
        else:
            speak('Good evening')

    wish()
    if flag == 1:
        speak('Where are you now?')
        query = takecommand().lower()
        return query
    else:
        speak('Where do you want to go?')
        query = takecommand().lower()
        return query


def execPresentation2(Fname):
    run_pygame(Fname)


def voice(flag=1):
    # Initialize COM explicitly
    comtypes.CoInitialize()

    # Initialize pyttsx3 engine
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Fix typo ('voices' → 'voice')
    engine.setProperty('rate', 150)  # Adjust speed (optional)
    engine.setProperty('volume', 1.0)  # Ensure max volume

    def speak(audio):
        print(f"Speaking: {audio}")  # Debug print
        engine.say(audio)
        engine.runAndWait()

    # Convert voice to text
    def takecommand():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print('Listening...')
            r.pause_threshold = 1
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                speak('Timeout occurred. Please try again.')
                return 'none'

        try:
            print('Recognizing...')
            query = r.recognize_google(audio, language='en-in')
            print(f'User said: {query}')
        except Exception as e:
            speak('Sorry, I could not understand that. Please try again.')
            return 'none'
        return query

    # Greeting function
    def wish():
        hour = int(datetime.datetime.now().hour)
        if hour >= 0 and hour <= 12:
            speak('Good morning')
        elif hour >= 12 and hour <= 18:
            speak('Good Afternoon')
        else:
            speak('Good evening')

    wish()

    if flag == 1:
        speak('Where are you now?')
        query = takecommand().lower()
    else:
        speak('Where do you want to go?')
        query = takecommand().lower()

    return query


import pygame
import threading
import glob
import os
from time import sleep


def run_pygame(Fname):
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption('Car Simulation')

    # Set up the window
    window = pygame.display.set_mode((1200, 400))

    # Load the track and car images
    track = pygame.image.load(Fname)  # Replace with your track image filename
    car = pygame.image.load('tesla.png')
    car = pygame.transform.scale(car, (30, 60))
    prev_direction = ''
    direction_count = 0
    # Car initial position
    car_x = 997
    car_y = 300

    # Other variables
    clock = pygame.time.Clock()
    focal_dis = 30
    direction = 'up'
    drive = True
    cam_x_offset = 0
    cam_y_offset = 0
    clicked_points = []  # List to store clicked points on the path
    end_point = (441, 60)  # Replace with actual end point coordinates

    def is_on_path(click_pos):
        track_rect = track.get_rect()
        return track_rect.collidepoint(click_pos)

    DELAY = 3
    counter = 1

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        drive = not drive
                        if counter % 2 == 1:
                            counter += 1
                        else:
                            counter += 1
                    elif event.key == pygame.K_r:
                        clicked_points = []
                    elif event.key == pygame.K_q:
                        if direction == 'left':
                            direction = 'right'
                            car = pygame.transform.rotate(car, 180)
                            cam_x_offset = 30
                        elif direction == 'down':
                            direction = 'up'
                            car = pygame.transform.rotate(car, 180)
                            cam_y_offset = -10

                if event.type == pygame.MOUSEBUTTONDOWN:
                    click_pos = pygame.mouse.get_pos()
                    if is_on_path(click_pos):
                        clicked_points.append(click_pos)

            if not drive:
                continue

            clock.tick(30)

            cam_x = car_x + 15 + cam_x_offset
            cam_y = car_y + 15 + cam_y_offset

            up_px = window.get_at((cam_x, cam_y - focal_dis))
            right_px = window.get_at((cam_x + focal_dis, cam_y))
            down_px = window.get_at((cam_x, cam_y + focal_dis))
            left_px = window.get_at((cam_x - focal_dis, cam_y))

            # Print the pixel values for debugging
            print(f"up_px: {up_px}, right_px: {right_px}, down_px: {down_px}, left_px: {left_px}")
            # Close the window if specific pixel values are detected
            if (up_px == (46, 170, 108, 255) and
                    right_px == (233, 194, 168, 255) and
                    down_px == (46, 170, 108, 255) and
                    left_px == (235, 51, 36, 255)):
                pygame.quit()
                return

            if (up_px == (172, 116, 98, 255) and
                    right_px == (46, 170, 108, 255) and
                    down_px == (237, 28, 36, 255) and
                    left_px == (46, 170, 108, 255)):
                pygame.quit()
                return

            # Check if the color is red
            if up_px[:3] == (255, 0, 0) or right_px[:3] == (255, 0, 0) or down_px[:3] == (255, 0, 0) or left_px[:3] == (
            255, 0, 0):
                drive = False

            if direction == 'up' and up_px[0] != 255 and right_px[0] == 255:
                direction = 'right'
                cam_x_offset = 30
                car = pygame.transform.rotate(car, -90)
                sleep(DELAY)
            elif direction == 'right' and right_px[0] != 255 and down_px[0] == 255:
                direction = 'down'
                car_x += 30
                cam_x_offset = 0
                cam_y_offset = 30
                car = pygame.transform.rotate(car, -90)
                sleep(DELAY)
            elif direction == 'down' and down_px[0] != 255 and right_px[0] == 255:
                direction = 'right'
                car_y += 30
                cam_x_offset = 30
                cam_y_offset = 0
                car = pygame.transform.rotate(car, 90)
                sleep(DELAY)
            elif direction == 'right' and right_px[0] != 255 and up_px[0] == 255:
                direction = 'up'
                car_x += 30
                cam_x_offset = 0
                car = pygame.transform.rotate(car, 90)
                sleep(DELAY)
            elif direction == 'up' and up_px[0] != 255 and left_px[0] == 255:
                direction = 'left'
                car = pygame.transform.rotate(car, 90)
                sleep(DELAY)
            elif direction == 'left' and left_px[0] != 255 and down_px[0] == 255:
                direction = 'down'
                cam_y_offset = 30
                car = pygame.transform.rotate(car, 90)
                sleep(DELAY)
            elif direction == 'down' and down_px[0] != 255 and left_px[0] == 255:
                direction = 'left'
                car_y += 30
                cam_y_offset = 0
                car = pygame.transform.rotate(car, -90)
                sleep(DELAY)
            elif direction == 'left' and left_px[0] != 255 and up_px[0] == 255:
                direction = 'up'
                car = pygame.transform.rotate(car, -90)
                sleep(DELAY)

            # Initialize COM explicitly
            comtypes.CoInitialize()

            # Initialize pyttsx3 engine
            engine = pyttsx3.init('sapi5')
            voices = engine.getProperty('voices')
            engine.setProperty('voices', voices[0].id)

            def speak(audio):
                engine.say(audio)
                engine.runAndWait()

            if direction != prev_direction:
                speak(direction)
                prev_direction = direction
                direction_count = 0
            else:
                direction_count += 1
                if direction_count == 30:
                    speak(direction)
                    direction_count = 0

            def check_collision(x, y):
                for rect in clicked_points:
                    if rect[0] < x < rect[0] + 10 and rect[1] < y < rect[1] + 10:
                        return True
                return False

            if direction == 'up' and up_px[0] == 255 and not check_collision(car_x, car_y - 2):
                car_y -= 2
            elif direction == 'right' and right_px[0] == 255 and not check_collision(car_x + 2, car_y):
                car_x += 2
            elif direction == 'down' and down_px[0] == 255 and not check_collision(car_x, car_y + 2):
                car_y += 2
            elif direction == 'left' and left_px[0] == 255 and not check_collision(car_x - 2, car_y):
                car_x -= 2
            elif left_px[0] != 255 and right_px[0] != 255 and up_px[0] != 255 and down_px[0] != 255:
                drive2 = False

            window.fill((0, 0, 0))
            window.blit(track, (0, 0))
            window.blit(car, (car_x, car_y))
            pygame.draw.circle(window, (0, 255, 0), (cam_x, cam_y), 5, 5)

            for point in clicked_points:
                pygame.draw.rect(window, (255, 0, 0), (point[0], point[1], 50, 50))

            pygame.display.update()
    finally:
        pygame.quit()
        return jsonify({"status": "Path Navigated Successfully"})


@app.route('/pathnavigator/start', methods=['POST'])
def voiceGuide():
    source = voice(1)
    source = "default"
    destination = voice(2)
    destination = "bob"
    res = source + '\\' + destination + '.png'
    execPresentation2(res)
    return jsonify({'status': 'Success'})


if __name__ == '__main__':
    app.run(debug=True)
