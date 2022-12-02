from flask import Flask, render_template, Response
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
# for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    #function about gen_frame

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # used to record the time when we processed last frame 
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame 
    new_frame_time = 0

    # Load Model
    with open("model/model.sav", 'rb') as file:
        model = pickle.load(file)

    hasil =[]
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        image = cv2.flip(frame, 1)
        if not success:
            break
        else:
            with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
                # Recolor Feed
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                try:
                    #Pose Detections
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    # Concate rows
                    pose = results.pose_landmarks.landmark
                    centering = []
                    pose_row = []
                    n=0

                    for landmark in pose:
                        x = list(np.array([landmark.x, landmark.y, landmark.z, landmark.visibility]).flatten())
                        for y in x:
                            pose_row.append(y)
                            if y == x[0]:
                                a = list(np.array([pose[0].x - y]))
                                for b in a:
                                    centering.append(b)
                            if y == x[1]:
                                a = list(np.array([pose[0].y - y]))
                                for b in a:
                                    centering.append(b)
                            if y == x[2]:
                                a = list(np.array([pose[0].z - y]).flatten())
                                for b in a:
                                    centering.append(b)
                    row = centering
                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]

                    urutan = ['Berdiri', 'Rukuk', 'Itidal', 'Sujud', 'Takhiyat Akhir', 'Sujud']
                    rakaat = 0
                    gerakan = 6
                    jml = 5
                    n = 0
                        
                    hasil.append(body_language_class)
                    for i in range(len(hasil)):
                        if i <= jml:
                            if hasil[i] == urutan[n]:
                                if i <= jml:
                                    gerakan -= 1
                                    jml += 5
                                    n+=1
                                    if gerakan == 0:
                                        rakaat += 1
                                        gerakan += 6
                                        n -= 6
                            else:
                                jml += 1
                                
                    # Get status box
                    cv2.rectangle(image, (0,0), (220, 180), (245, 117, 16), -1)
                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (15,132), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, body_language_class
                                , (10,163), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROBABILITY'
                                , (15,24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,53), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
                    # Display Probability
                    cv2.putText(image, 'RAKAAT'
                                , (15,76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(rakaat)
                                , (10,107), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                except Exception as e:
                    continue

                finally:           
                    # font which we will be using to display FPS
                    font = cv2.FONT_HERSHEY_SIMPLEX

                        # time when we finish processing for this frame
                    new_frame_time = time.time()

                    fps = 1 / (new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time

                        # converting the fps into integer
                    fps = int(round(fps))

                        # converting the fps to string so that we can display it on frame
                        # by using putText function
                    fps = str(fps)

                        # puting the FPS count on the frame
                    cv2.putText(image, fps, (550, 50), font, 2, (100, 255, 0), 3, cv2.LINE_AA)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                    ret, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)