import cv2
import mediapipe as mp
import math

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    # Calculate vectors
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    # Calculate dot product and magnitudes
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    # Calculate angle in radians and convert to degrees
    angle_radians = math.acos(dot_product / (magnitude_ab * magnitude_bc))
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

# Setup video capture
#video_file = "pose1.mp4"
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame color from BGR to RGB for processing
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True

    # Convert back to BGR for display
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        h, w, _ = frame.shape

        # Get right shoulder coordinates
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        cx_rs, cy_rs = int(right_shoulder.x * w), int(right_shoulder.y * h)
        
        # Get right hip coordinates
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        cx_rh, cy_rh = int(right_hip.x * w), int(right_hip.y * h)
        
        # Get right ear coordinates
        right_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        cx_re, cy_re = int(right_ear.x * w), int(right_ear.y * h)

        # Calculate upper points (green)
        offset = 60
        upper_shoulder = (cx_rs, max(0, cy_rs - offset))
        upper_hip = (cx_rh, max(0, cy_rh - offset))

        # Draw green dots
        cv2.circle(frame, upper_shoulder, 5, (0, 255, 0), -1)
        cv2.circle(frame, upper_hip, 5, (0, 255, 0), -1)

        # Draw lines
        cv2.line(frame, (cx_rh, cy_rh), (cx_rs, cy_rs), (255, 0, 255), 2)  # Hip to shoulder
        cv2.line(frame, (cx_rs, cy_rs), (cx_re, cy_re), (255, 255, 0), 2)  # Shoulder to ear
        cv2.line(frame, (cx_rh, cy_rh), upper_hip, (0, 165, 255), 2)       # Hip to upper hip
        cv2.line(frame, (cx_rs, cy_rs), upper_shoulder, (0, 255, 255), 2)  # Shoulder to upper shoulder

        # Calculate angles
        angle_hip = calculate_angle(upper_hip, (cx_rh, cy_rh), (cx_rs, cy_rs))  # Hip posture angle
        angle_neck = calculate_angle((cx_rs, cy_rs), (cx_re, cy_re), upper_shoulder)  # Neck posture angle

        # Determine posture
        hip_posture = "Good" if 160 <= angle_hip <= 180 else "Poor"
        neck_posture = "Good" if 150 <= angle_neck <= 180 else "Poor"
        hip_color = (0, 255, 0) if hip_posture == "Good" else (0, 0, 255)  # Green if Good, Red if Poor
        neck_color = (0, 255, 0) if neck_posture == "Good" else (0, 0, 255)  # Green if Good, Red if Poor

        # Display angles and posture with dynamic colors
        cv2.putText(frame, f"Hip Angle: {angle_hip:.1f} ({hip_posture})", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, hip_color, 2)
        cv2.putText(frame, f"Neck Angle: {angle_neck:.1f} ({neck_posture})", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, neck_color, 2)

    # Overlay the FPS value
    cv2.putText(frame, f"FPS: {fps}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video Output', frame)

    # Exit loop when 'Esc' key is pressed
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()