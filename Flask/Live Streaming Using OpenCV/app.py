import os
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Get the absolute path to the Haarcascades directory
haarcascades_dir = os.path.join(os.path.dirname(cv2.__file__), 'data')

# Initialize detectors
face_cascade_path = os.path.join(haarcascades_dir, 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(haarcascades_dir, 'haarcascade_eye.xml')

face_detector = cv2.CascadeClassifier(face_cascade_path)
eye_detector = cv2.CascadeClassifier(eye_cascade_path)

def gen_frames():
    if face_detector.empty():
        raise ValueError("Failed to load face cascade classifier")
    if eye_detector.empty():
        raise ValueError("Failed to load eye cascade classifier")
    
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.1, 7)
                
                # Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    eyes = eye_detector.detectMultiScale(roi_gray, 1.1, 3)
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error in face detection: {str(e)}")
                # You might want to yield a default frame or error message here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    app.run(debug=True)
