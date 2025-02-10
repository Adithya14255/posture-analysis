import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between two points and vertical axis
def calculate_vertical_angle(a, b, vertical_offset=0):
    a = np.array(a)
    b = np.array(b)

    # Adjust vertical reference point with optional offset
    vertical_ref = np.array([a[0] + vertical_offset, a[1] - 100])

    # Calculate angle with vertical
    radians = np.arctan2(b[1] - a[1], b[0] - a[0]) - np.arctan2(vertical_ref[1] - a[1], vertical_ref[0] - a[0])
    angle = np.abs(np.degrees(radians))

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make pose detection
    results = pose.process(image)

    # Convert the image back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for neck posture using the leftmost detected point
        leftmost_point = min(landmarks, key=lambda lm: lm.x)
        leftmost_coord = [leftmost_point.x * frame.shape[1], leftmost_point.y * frame.shape[0]]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]

        # Calculate neck and back angles relative to vertical
        neck_angle = calculate_vertical_angle(left_shoulder, leftmost_coord)
        back_angle = calculate_vertical_angle(left_hip, left_shoulder, vertical_offset=10)  # Slight incline adjustment for back

        # Display angles
        cv2.putText(image, f'Neck Incline: {int(neck_angle)} deg', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Back Incline: {int(back_angle)} deg', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Posture feedback for neck
        if neck_angle < 10:
            neck_feedback = "Good Neck Posture"
            neck_color = (0, 255, 0)
        else:
            neck_feedback = "Adjust Your Neck"
            neck_color = (0, 0, 255)

        # Posture feedback for back
        if back_angle < 10:
            back_feedback = "Good Back Posture"
            back_color = (0, 255, 0)
        else:
            back_feedback = "Adjust Your Back"
            back_color = (0, 0, 255)

        # Display feedback
        cv2.putText(image, neck_feedback, (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, neck_color, 2, cv2.LINE_AA)
        cv2.putText(image, back_feedback, (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, back_color, 2, cv2.LINE_AA)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the output
    cv2.imshow('Ergonomic Posture Estimator', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
