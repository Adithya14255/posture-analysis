from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Global variables to store posture data
posture_data = {'neck_angle': 0, 'back_angle': 0, 'neck_feedback': '', 'back_feedback': ''}

# Function to calculate the angle between two points and vertical axis
def calculate_vertical_angle(a, b, vertical_offset=0):
    a = np.array(a)
    b = np.array(b)

    vertical_ref = np.array([a[0] + vertical_offset, a[1] - 100])

    radians = np.arctan2(b[1] - a[1], b[0] - a[0]) - np.arctan2(vertical_ref[1] - a[1], vertical_ref[0] - a[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Video capture generator
def gen():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            leftmost_point = min(landmarks, key=lambda lm: lm.x)
            leftmost_coord = [leftmost_point.x * frame.shape[1], leftmost_point.y * frame.shape[0]]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]

            neck_angle = calculate_vertical_angle(left_shoulder, leftmost_coord)
            back_angle = calculate_vertical_angle(left_hip, left_shoulder, vertical_offset=10)

            posture_data['neck_angle'] = int(neck_angle)
            posture_data['back_angle'] = int(back_angle)
            posture_data['neck_feedback'] = "Good Neck Posture" if neck_angle < 10 else "Adjust Your Neck"
            posture_data['back_feedback'] = "Good Back Posture" if back_angle < 10 else "Adjust Your Back"
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data_feed')
def data_feed():
    return jsonify(posture_data)

if __name__ == '__main__':
    app.run(debug=True)
