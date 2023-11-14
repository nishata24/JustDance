import cv2
import mediapipe as mp
import sys

# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from file.
video_path = sys.argv[1]  # Get video file path from command line argument
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save the output.
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use 'avc1' (H.264 codec)
out = cv2.VideoWriter('move3_annotated.mp4', fourcc, frame_rate, (frame_width, frame_height))

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
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Write the frame into the output video file
    out.write(frame)

# Release resources.
pose.close()
cap.release()
out.release()
