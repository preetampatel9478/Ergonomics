# Ergonomics
import cv2
import math as m
import mediapipe as mp
from plyer import notification
# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi * theta)
    return degree

"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""
def sendWarning():
    notification_title = "Posture Alert"
    notification_text = "You are sitting in wrong posture"
    notification_timeout = 5
    notification.notify(
        title = notification_title, message = notification_text, timeout = notification_timeout
    )
# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initialize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#

if __name__ == "__main__":
    # For webcam input replace file name with 0.
    file_name = "C:\\Users\\saura\\OneDrive\\Desktop\\minor project 2\\input1.mp4"
    cap = cv2.VideoCapture(file_name)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output1.mp4', fourcc, fps, frame_size)

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        if lm is not None:
            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Left elbow.
            l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
            l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
            # Right elbow.
            r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
            r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)

            # Left wrist.
            l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
            l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
            # Right wrist.
            r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
            r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)

            # Left knee.
            l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
            l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
            # Right knee.
            r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
            r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)

            # Left ankle.
            l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
            l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
            # Right ankle.
            r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
            r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            if offset < 100:
                cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.6, green, 1)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.6, red, 1)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            left_arm_angle = findAngle(l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)
            right_arm_angle = findAngle(r_shldr_x, r_shldr_y, r_elbow_x, r_elbow_y)

            # Draw landmarks.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

            cv2.circle(image, (l_elbow_x, l_elbow_y), 7, yellow, -1)
            cv2.circle(image, (r_elbow_x, r_elbow_y), 7, yellow, -1)
            cv2.circle(image, (l_wrist_x, l_wrist_y), 7, yellow, -1)
            cv2.circle(image, (r_wrist_x, r_wrist_y), 7, yellow, -1)

            cv2.circle(image, (l_knee_x, l_knee_y), 7, blue, -1)
            cv2.circle(image, (r_knee_x, r_knee_y), 7, blue, -1)
            cv2.circle(image, (l_ankle_x, l_ankle_y), 7, blue, -1)
            cv2.circle(image, (r_ankle_x, r_ankle_y), 7, blue, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
            cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

            # Similarly, here we are taking y - coordinate 100px above x1. Note that
            # you can take any value for y, not necessarily 100 or 200 pixels.
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)

            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + ' Torso : ' + str(int(torso_inclination))
            left_arm_text_string = 'Left Arm : ' + str(int(left_arm_angle))
            right_arm_text_string = 'Right Arm : ' + str(int(right_arm_angle))

            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            if neck_inclination < 40 and torso_inclination < 10:
                bad_frames = 0
                good_frames += 1

                cv2.putText(image, angle_text_string, (10, 30), font, 0.6, light_green, 1)
                cv2.putText(image, left_arm_text_string, (10, 60), font, 0.6, light_green, 1)
                cv2.putText(image, right_arm_text_string, (500, 60), font, 0.6, light_green, 1)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.6, light_green, 1)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.6, light_green, 1)
                cv2.putText(image, str(int(left_arm_angle)), (l_elbow_x + 10, l_elbow_y), font, 0.6, light_green, 1)
                cv2.putText(image, str(int(right_arm_angle)), (r_elbow_x + 10, r_elbow_y), font, 0.6, light_green, 1)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)


            else:
                good_frames = 0
                bad_frames += 1

                cv2.putText(image, angle_text_string, (10, 30), font, 0.6, red, 1)
                cv2.putText(image, left_arm_text_string, (10, 60), font, 0.6, red, 1)
                cv2.putText(image, right_arm_text_string, (500, 60), font, 0.6, red, 1)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.6, red, 1)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.6, red, 1)
                cv2.putText(image, str(int(left_arm_angle)), (l_elbow_x + 10, l_elbow_y), font, 0.6, red, 1)
                cv2.putText(image, str(int(right_arm_angle)), (r_elbow_x + 10, r_elbow_y), font, 0.6, red, 1)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

            # Calculate the time of remaining in a particular posture.
            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            # Pose time.
            if good_time > 0:
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                cv2.putText(image, time_string_good, (10, h - 20), font, 0.6, green, 1)
            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                cv2.putText(image, time_string_bad, (10, h - 20), font, 0.6, red, 1)

            # If you stay in bad posture for more than 3 minutes (180s) send an alert.
            if bad_time > 10:
                sendWarning()
        # Write frames.
        video_output.write(image)

        # Display.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
