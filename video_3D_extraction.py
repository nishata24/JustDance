#3d landmarks extraction from pre-recorded video
#33 landmarks (0 to 32)
import cv2
import mediapipe as mp
import sys
import csv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the Pose model with model_complexity=2 for 3D landmarks.
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(sys.argv[1])

if not cap.isOpened():
    print("Error opening video stream or file")
    raise TypeError

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

outdir, inputflnm = sys.argv[1][:sys.argv[1].rfind('/')+1], sys.argv[1][sys.argv[1].rfind('/')+1:]
inflnm, inflext = inputflnm.split('.')
out_filename = f'{outdir}{inflnm}_annotated.{inflext}'
out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width, frame_height))

# Open a CSV file to save the landmarks in world coordinates
csv_file = open('landmarks_world_coordinates.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Landmark", "x", "y", "z"])  # Write header

frame_count = 0
while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    frame_count += 1

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    if results.pose_landmarks:
        # Save the world coordinates to the CSV file
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            x, y, z = landmark.x, landmark.y, landmark.z  # These are the world coordinates.
            csv_writer.writerow([frame_count, idx, x, y, z])

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    out.write(image)

pose.close()
cap.release()
out.release()
csv_file.close()

