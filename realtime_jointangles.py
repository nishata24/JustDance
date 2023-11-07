#landmarks
import cv2
import mediapipe as mp
import numpy as np

# Define a function to calculate the angle between three points in 3D.
def calculate_angle_3d(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    # Create vectors from points
    ba = a - b
    bc = c - b
    
    # Calculate the cosine angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    # Convert to degrees
    angle = np.degrees(angle)
    
    return angle

# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from file or camera.
cap = cv2.VideoCapture(0)  # Replace 0 with video file path if needed.

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the frame to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose.
    results = pose.process(frame_rgb)

    # Draw the pose annotation on the frame.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmarks in 3D.
        landmarks = results.pose_landmarks.landmark
        
        # You can use the visibility (v value) to filter out low-confidence landmarks
        # For simplicity, this example does not do that.

        # Calculate and display angles for both shoulders, elbows, and knees.
        # Right elbow angle in 3D
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, 
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, 
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
        angle_right_elbow = calculate_angle_3d(shoulder_right, elbow_right, wrist_right)
        # Since we can't plot text in 3D, we're using the elbow's x and y for text placement
        cv2.putText(frame, str(int(angle_right_elbow)), 
                    (int(elbow_right[0] * frame.shape[1]), int(elbow_right[1] * frame.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Similarly calculate angles for left elbow, right knee, and left knee using the 3D coordinates
        # ...

    # Display the frame.
    cv2.imshow('MediaPipe Pose with 3D Angle Calculation', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Release the capture and close windows.
cap.release()
cv2.destroyAllWindows()
