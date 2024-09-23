from flask import Flask, redirect, url_for, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app = Flask(__name__)


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = round(np.abs(radians*180.0/np.pi), 0)

    if angle > 180.0:
        angle = 360-angle
    return round(angle, 0)


def curls_frame():
    cap = cv2.VideoCapture(0)
    l_counter = 0
    r_counter = 0
    l_stage = None
    r_stage = None
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            fframe = cv2.flip(frame, 1)
            image = cv2.cvtColor(fframe, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                cv2.putText(image, str(l_angle),
                            tuple(np.multiply(
                                l_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                if l_angle > 170:
                    l_stage = "down"
                if l_angle < 15 and l_stage == 'down':
                    l_stage = "up"
                    l_counter += 1

                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                cv2.putText(image, str(r_angle),
                            tuple(np.multiply(
                                r_elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                if r_angle > 170:
                    r_stage = "down"
                if r_angle < 15 and r_stage == 'down':
                    r_stage = "up"
                    r_counter += 1

            except:
                pass
            cv2.rectangle(image, (0, 385), (70, 480), (255, 255, 255), -1)
            cv2.putText(image, 'LEFT', (6, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'REPS', (7, 395),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(r_counter),
                        (6, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (245, 117, 16), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (8, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, r_stage,
                        (7, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 117, 16), 1, cv2.LINE_AA)
            cv2.rectangle(image, (570, 385), (640, 480), (255, 255, 255), -1)

            cv2.putText(image, 'RIGHT', (575, 380),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'REPS', (580, 395),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(l_counter),
                        (570, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (245, 117, 16), 2, cv2.LINE_AA)

            cv2.putText(image, 'STAGE', (580, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, l_stage,
                        (580, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 117, 16), 1, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
            cv2.imwrite('t.jpg', image)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
    cap.release()


@app.route("/")
def index():
    return render_template("curl.html")


@app.route("/curls")
def curls():
    return render_template("index.html")



@app.route("/curls_feed")
def curls_feed():
    return Response(curls_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()
