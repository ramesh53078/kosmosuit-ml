import cv2
import mediapipe as mp
from flask import Flask, render_template, Response
import SquatPosture as sp
from utils import *

from flask import Flask

app = Flask(__name__, template_folder='.')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

activity_status = "Idle"  # Default status

# return 'Hello, World!'

def classify_activity(coords):
    global activity_status

    nose_y = coords[mp_pose.PoseLandmark.NOSE.value][1]
    left_ankle_y = coords[mp_pose.PoseLandmark.LEFT_ANKLE.value][1]
    right_ankle_y = coords[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1]

    # Set thresholds for different activities
    jump_threshold = 20
    walking_threshold = 5  # Threshold for distinguishing walking
    running_threshold = 15  # Threshold for distinguishing running

    print(f"Nose Y: {nose_y}, Left Ankle Y: {left_ankle_y}, Right Ankle Y: {right_ankle_y}")

    if nose_y > left_ankle_y + jump_threshold and nose_y > right_ankle_y + jump_threshold:
        activity_status = "Jumping"
    elif abs(left_ankle_y - right_ankle_y) > running_threshold:
        activity_status = "Running"
    elif abs(left_ankle_y - right_ankle_y) > walking_threshold:
        activity_status = "Walking"
    else:
        activity_status = "Idle"

    return activity_status

def generate_frames():
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while True:
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            params = sp.get_params(results, all=True)
            print(params)

            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            coords = landmarks_list_to_array(results.pose_landmarks, frame.shape)

            # Classify activity and update status
            activity_status = classify_activity(coords)

            # Display activity status on the frame
            cv2.putText(frame, f"Activity: {activity_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Use Flask's `stream_with_context` to include JavaScript in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                   b'<script>document.getElementById("activityStatus").innerText = "{activity_status}";</script>\r\n')

# @app.route('/kosmosuit-ml/')
# def hello():
#     return 'Hello, World!'

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
        # return 'Hello, World!'
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

