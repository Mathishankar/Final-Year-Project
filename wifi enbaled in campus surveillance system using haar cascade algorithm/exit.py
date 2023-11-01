import cv2
import os
from flask import Flask,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import pygame

#### Defining Flask App
app = Flask(__name__)
port = 5001




#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%d_%m_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")




#### Initializing VideoCapture object to access WebCam
face_detector_default = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
face_detector_alt = cv2.CascadeClassifier('static/haarcascade_frontalface_alt.xml')
cap2 = cv2.VideoCapture()

if not cap2.isOpened():
    print("Error opening cameras")
    exit()

#### If these directories don't exist, create them
if not os.path.isdir('Exit'):
    os.makedirs('Exit')

    
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Exit-{datetoday}.csv' not in os.listdir('Exit'):
    with open(f'Exit/Exit-{datetoday}.csv','w') as f:
        f.write('Name,Id,ExitTime')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector_default.detectMultiScale(gray, 1.3, 5)
    face_points = face_detector_default.detectMultiScale(gray, 1.3, 5)   
    return face_points


#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (20, 20))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's Exit file in Exit folder
def extract_Exit():
    df = pd.read_csv(f'Exit/Exit-{datetoday}.csv')
    names = df['Name']
    rolls = df['Id']
    times = df['ExitTime']
    l = len(df)
    return names,rolls,times,l


#### Add Exit of a specific user
def add_Exit(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Exit/Exit-{datetoday}.csv')
    if int(userid) not in list(df['Id']):
        with open(f'Exit/Exit-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def exit():
    names,rolls,times,l = extract_Exit()    
    return render_template('exit.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we click on Take Exit Button
@app.route('/start',methods=['GET'])
def start():
    # Check if face recognition model exists
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('exit.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.')
    
    # Initialize Pygame and buzzer sound
    pygame.init()
    buzzer_sound = pygame.mixer.Sound("D:/OneDrive/Desktop/New folder/buzzer.wav")

    # Open video capture device
    cap = cv2.VideoCapture(1)
    ret2= True

    # Loop through frames
    while ret2:
        # Read a frame
        ret2, frame = cap.read()

        # Extract faces from frame
        faces = extract_faces(frame)

        # Loop through detected faces
        for face in faces:
            # Get coordinates of face
            (x, y, w, h) = face

            # Extract face image and resize it
            face_img = cv2.resize(frame[y:y+h,x:x+w], (20, 20))

            # Identify the person in the face image
            identified_person = identify_face(face_img.reshape(1,-1))[0]

            # If the person is not in the database, play buzzer sound
            if identified_person is None:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                buzzer_sound.play().sleep(5)
            # If the person is in the database, mark their exit and display their name
            else:
                add_Exit(identified_person)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'{identified_person}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Exit', frame)

        # Break out of loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release video capture device and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    # Extract data from exit log and render exit page
    names, rolls, times, l = extract_Exit()    
    return render_template('exit.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#### Our main function which runs the Flask App
if __name__ == '__main__':
    os.environ.setdefault('FLASK_DEBUG', 'development')
    app.run(port=port,debug=True)