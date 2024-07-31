import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import speech_recognition as sr
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to find the midpoint between two points
def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)

# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Define thresholds for detecting eye closure
EAR_THRESHOLD = 0.25
CLOSED_EYE_TIME = 2.0  # Time in seconds to detect a blink

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the previous position of the eyes and time stamps for blink detection
prev_left_pupil = None
prev_right_pupil = None
left_eye_closed_time = 0
right_eye_closed_time = 0

# Initialize smoothing parameters
smooth_factor = 0.5
smooth_dx = 0
smooth_dy = 0

# Scaling factor for mouse movement sensitivity
sensitivity_factor = 3.0
speed_display = f"Speed: {sensitivity_factor:.1f}"

# Voice command handler
def handle_voice_command():
    global sensitivity_factor, speed_display
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = None
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("Listening timed out, retrying...")
        if audio is not None:
            try:
                command = recognizer.recognize_google(audio).lower()
                if "speed up" in command:
                    sensitivity_factor += 1.0
                elif "speed down" in command:
                    sensitivity_factor -= 1.0
                sensitivity_factor = max(1.0, sensitivity_factor)  # Prevent negative or zero speed
                speed_display = f"Speed: {sensitivity_factor:.1f}"
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
            except Exception as e:
                print(f"Error: {e}")

# Initialize face mesh
with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    last_command_check = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find face landmarks
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmark coordinates
                landmarks = [(int(point.x * frame.shape[1]), int(point.y * frame.shape[0]))
                             for point in face_landmarks.landmark]

                # Define eye landmark indices in MediaPipe Face Mesh
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                left_eye_points = [landmarks[i] for i in left_eye_indices]
                right_eye_points = [landmarks[i] for i in right_eye_indices]

                # Calculate Eye Aspect Ratios
                left_ear = calculate_ear(left_eye_points)
                right_ear = calculate_ear(right_eye_points)

                current_time = time.time()

                # Check for eye closure
                if left_ear < EAR_THRESHOLD:
                    if left_eye_closed_time == 0:
                        left_eye_closed_time = current_time
                    elif current_time - left_eye_closed_time >= CLOSED_EYE_TIME:
                        pyautogui.click(button='left')
                        left_eye_closed_time = 0
                else:
                    left_eye_closed_time = 0

                if right_ear < EAR_THRESHOLD:
                    if right_eye_closed_time == 0:
                        right_eye_closed_time = current_time
                    elif current_time - right_eye_closed_time >= CLOSED_EYE_TIME:
                        pyautogui.click(button='right')
                        right_eye_closed_time = 0
                else:
                    right_eye_closed_time = 0

                # Define indices for eye corner landmarks
                left_eye_corner_indices = [133, 173]
                right_eye_corner_indices = [362, 398]

                left_eye_corners = [landmarks[i] for i in left_eye_corner_indices]
                right_eye_corners = [landmarks[i] for i in right_eye_corner_indices]

                # Calculate the pupil center as the midpoint between the eye corners
                left_pupil = midpoint(left_eye_corners[0], left_eye_corners[1])
                right_pupil = midpoint(right_eye_corners[0], right_eye_corners[1])

                if prev_left_pupil is not None and prev_right_pupil is not None:
                    # Calculate movement in both pupils
                    dx = (left_pupil[0] - prev_left_pupil[0] + right_pupil[0] - prev_right_pupil[0]) // 2
                    dy = (left_pupil[1] - prev_left_pupil[1] + right_pupil[1] - prev_right_pupil[1]) // 2

                    # Apply smoothing
                    smooth_dx = smooth_factor * smooth_dx + (1 - smooth_factor) * dx
                    smooth_dy = smooth_factor * smooth_dy + (1 - smooth_factor) * dy

                    # Scale the movement for sensitivity
                    move_x = int(smooth_dx * sensitivity_factor)
                    move_y = int(smooth_dy * sensitivity_factor)

                    # Move mouse based on the average pupil movement
                    pyautogui.moveRel(move_x, move_y)

                prev_left_pupil = left_pupil
                prev_right_pupil = right_pupil

                # Draw the pupil centers
                cv2.circle(frame, left_pupil, 3, (0, 255, 0), -1)
                cv2.circle(frame, right_pupil, 3, (0, 255, 0), -1)
                eye_center = midpoint(left_pupil, right_pupil)
                cv2.circle(frame, eye_center, 3, (0, 0, 255), -1)

                # Display the speed on the frame
                cv2.putText(frame, speed_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check for voice commands every 15 seconds
        if time.time() - last_command_check > 15:
            handle_voice_command()
            last_command_check = time.time()

cap.release()
cv2.destroyAllWindows()
