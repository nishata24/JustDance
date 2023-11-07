#world landmarks
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

# Initialize MediaPipe Pose with model complexity 1 to enable 3D landmarks.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

        # Check for world landmarks.
        if results.pose_world_landmarks:
            # Get world landmarks.
            world_landmarks = results.pose_world_landmarks.landmark

            # Calculate and display angles for both shoulders, elbows, and knees in 3D.
            # Right elbow angle in 3D
            shoulder_right = [world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                              world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, 
                              world_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            elbow_right = [world_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                           world_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, 
                           world_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            wrist_right = [world_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                           world_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, 
                           world_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
            angle_right_elbow = calculate_angle_3d(shoulder_right, elbow_right, wrist_right)

            # Display angle. Convert from normalized space to image space.
            # ...
            elbow_right_image_space = [elbow_right[0] * frame.shape[1], elbow_right[1] * frame.shape[0]]

            cv2.putText(frame, str(int(angle_right_elbow)), 
            (int(elbow_right_image_space[0]), int(elbow_right_image_space[1])),
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
