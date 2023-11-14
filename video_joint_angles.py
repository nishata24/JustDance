import cv2
import mediapipe as mp
import numpy as np
import sys

# Define a function to calculate the angle between three points in 3D.
import numpy as np

def calculate_angle(P1, P2, P3):
    """
    Calculates the angle at point P1 formed by the line segments P1P2 and P1P3 using the atan2 method.

    Parameters:
    P1, P2, P3: Lists or tuples with two elements representing the x and y coordinates of points P1, P2, and P3, respectively.

    Returns:
    The angle at P1 in degrees within the range [0, 360).
    """
    # Calculate the angles using atan2
    angle = np.degrees(
        np.arctan2(P3[1] - P1[1], P3[0] - P1[0]) -
        np.arctan2(P2[1] - P1[1], P2[0] - P1[0])
    )

    # Normalize the angle to be within the range [0, 360) degrees
    angle = angle % 360

    return angle

# Example usage:
# Define the points as (x, y) pairs
P1 = (1, 2)
P2 = (2, 3)
P3 = (3, 2)

# Calculate the angle at P1
angle_at_P1 = calculate_angle(P1, P2, P3)
print(f"The angle at P1 is: {angle_at_P1} degrees")

# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from file or camera.
cap = cv2.VideoCapture(sys.argv[1])  # Replace 0 with video file path if needed.

# Prepare an array to store angle-time data.
angle_time_data = []

# Get the frame rate of the video
frame_rate = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Current timestamp based on frame rate and frames read
    frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
    timestamp = frame_id / frame_rate

    # Convert the frame to RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose.
    results = pose.process(frame_rgb)

    # Draw the pose annotation on the frame.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get landmarks in 3D and calculate angles as before...

        # Save the angle and the timestamp to the list
        

        # Rest of your code for drawing and displaying the frame...
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
        angle_right_elbow = calculate_angle(elbow_right, shoulder_right, wrist_right)
        #angle_time_data.append((timestamp, angle_right_elbow))
        # Left elbow angle in 3D
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, 
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, 
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, 
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
        angle_left_elbow = calculate_angle(elbow_left, shoulder_left, wrist_left)
        #angle_time_data.append((timestamp, angle_left_elbow))
        
        # Right knee angle in 3D
        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y, 
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
        knee_right = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, 
              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
        ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y, 
               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
        angle_right_knee = calculate_angle(knee_right, hip_right, ankle_right)
        #angle_time_data.append((timestamp, angle_right_knee))

        # Left knee angle in 3D
        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y, 
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
        knee_left = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y, 
             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y, 
              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
        angle_left_knee = calculate_angle(knee_left, hip_left, ankle_left)
        #angle_time_data.append((timestamp, angle_left_knee))
        
        # Right shoulder angle in 3D
        neck_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y - 0.1,  # Approximation for the neck position
              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
        elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, 
               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
        angle_right_shoulder = calculate_angle(shoulder_right, neck_right, elbow_right)
        #angle_time_data.append((timestamp, angle_right_shoulder))

        # Left shoulder angle in 3D
        neck_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y - 0.1,  # Approximation for the neck position
             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, 
                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
        elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, 
              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
        angle_left_shoulder = calculate_angle(shoulder_left, neck_left, elbow_left)
        angle_time_data.append((timestamp, angle_right_elbow, angle_left_elbow, angle_right_knee, angle_left_knee, angle_right_shoulder, angle_left_shoulder))
pose.close()
cap.release()
#csv_file.close()


# Save the angle-time data to a file
with open('angle_move3.csv', 'w') as file:
    file.write('timestamp, angle_right_elbow, angle_left_elbow, angle_left_knee, angle_right_knee, angle_right_shoulder, angle_left_shoulder\n')
    for data in angle_time_data:
        file.write(f'{data[0]},{data[1]}, {data[2]}, {data[3]}, {data[4]}, {data[5]}, {data[6]}\n')

print("Angle-time data has been saved to angle_move3.csv.")
