# Eye-Tracking Mouse Control with Voice Command Speed Adjustment

This project leverages computer vision and voice recognition to control mouse movements using eye tracking. The mouse movement is highly sensitive to eye movements, and voice commands are used to adjust the sensitivity of the mouse.

## Features

- **Eye Tracking**: Uses MediaPipe's Face Mesh to detect eye landmarks and calculate eye movements.
- **Mouse Control**: Moves the mouse cursor based on detected eye movements.
- **Blink Detection**: Detects eye blinks to trigger left and right mouse clicks.
- **Voice Command**: Allows adjustment of mouse sensitivity using voice commands ("speed up" and "speed down").

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- PyAutoGUI
- SpeechRecognition
- PyAudio

## Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/your-repo/eye-tracking-mouse-control.git
    cd eye-tracking-mouse-control
    ```

2. **Create a Virtual Environment** (optional but recommended):
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```sh
    pip install opencv-python-headless mediapipe pyautogui SpeechRecognition pyaudio
    ```

4. **Ensure you have a Microphone**: This is required for voice command functionality.

## Usage

1. **Run the Script**:
    ```sh
    python main.py
    ```

2. **Voice Commands**:
    - Say "speed up" to increase the mouse movement sensitivity.
    - Say "speed down" to decrease the mouse movement sensitivity.

3. **Mouse Control**:
    - Move your eyes to control the mouse cursor.
    - Close your left eye for 2 seconds to trigger a left-click.
    - Close your right eye for 2 seconds to trigger a right-click.
