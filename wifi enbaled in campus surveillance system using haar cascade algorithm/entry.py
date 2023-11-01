import cv2
import os
from flask import Flask,request,render_template
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import pygame


#### Defining Flask App
app = Flask(__name__)
port=4001




#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%d_%m_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector_default = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
face_detector_alt = cv2.CascadeClassifier('static/haarcascade_frontalface_alt.xml')
cap1 = cv2.VideoCapture(0)

if not cap1.isOpened():
    print("Error opening cameras")
    exit()


#### If these directories don't exist, create them
if not os.path.isdir('Entry'):
    os.makedirs('Entry')

    
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Entry-{datetoday}.csv' not in os.listdir('Entry'):
    with open(f'Entry/Entry-{datetoday}.csv','w') as f:
        f.write('Name,Id,EntryTime')


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


#### Extract info from today's Entry file in Entry folder
def extract_Entry():
    df = pd.read_csv(f'Entry/Entry-{datetoday}.csv')
    names = df['Name']
    rolls = df['Id']
    times = df['EntryTime']
    l = len(df)
    return names,rolls,times,l


#### Add Entry of a specific user
def add_Entry(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Entry/Entry-{datetoday}.csv')
    if int(userid) not in list(df['Id']):
        with open(f'Entry/Entry-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def entry():
    names,rolls,times,l = extract_Entry()    
    return render_template('entry.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we click on Take Entry Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('entry.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    # Initialize Pygame and buzzer sound
    pygame.init()
    buzzer_sound = pygame.mixer.Sound("D:/OneDrive/Desktop/New folder/buzzer.wav")

    cap1 = cv2.VideoCapture(0)
    ret1 = True
    while ret1:
                # Read a frame
        ret1, frame = cap1.read()
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
                add_Entry(identified_person)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
                cv2.putText(frame, f'{identified_person}', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

        cv2.imshow('Entry',frame)

        if cv2.waitKey(1)==27:
            break

    cap1.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_Entry()    
    return render_template('entry.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap1 = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame1 = cap1.read()
        faces = extract_faces(frame1)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame1,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame1,f'Images Captured: {i}/40',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame1[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==400:
            break
        cv2.imshow('Adding new User',frame1)
        if cv2.waitKey(1)==27:
            break
    cap1.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_Entry()    
    return render_template('entry.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### Our main function which runs the Flask App
if __name__ == '__main__':
    os.environ.setdefault('FLASK_DEBUG', 'development')
    app.run(port=port,debug=True)