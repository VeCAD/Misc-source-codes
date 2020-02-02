'''
Using opencv libraries/database 
The images are from Chokepoint dataset 
'''

import sys 
import cv2
import time
import numpy as np

cap = cv2.VideoCapture('input_video.mp4')

#using opencv haar classifier and facemark libs
#meant for frontal face but still works to a certain degree
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

#create facial landmark detector and load model
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")
lndmk=[] 

#face 3d model
model_points_3d = np.array([(0.0, 0.0, 0.0),             #nose tip
                            (0.0, -330.0, -65.0),        #chin
                            (-225.0, 170.0, -135.0),     #left eye corner 
                            (225.0, 170.0, -135.0),      #right eye corner 
                            (-150.0, -150.0, -125.0),    #left mouth corner
                            (150.0, -150.0, -125.0)],    #right mouth corner  
                            dtype="double")      
#camera distortion
cam_dist_coeffs = np.zeros((4,1)) 

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
#In this case visual marker will be subject's eyes, nose , mouth landmarks
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, im = cap.read()
  if ret == True:
    #scale down the image
    im = cv2.resize(im,(640,480))
    #OR scale up
    #im = cv2.resize(im,(1024,768))

    #print ("Scaled size {0}".format(im.shape))
    size = im.shape

    #convert to gray for haar classifier
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #############################
    #Face and landmark detection#
    #############################
    
    #detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #run landmark detection/post estimation only if face detected in frame
    FC = np.array(faces)
    if (FC.size>0):
        lndmk = facemark.fit(im, faces)
    
        #save detected landmarks to array
        F = np.array(lndmk[1])
    
    #######################
    ##Head pose estimation#
    #######################
        #pull out key face markers from detected face landmarks
        facial_markers = np.array([(F[0,0,30,0], F[0,0,30,1]),      #nose tip
                            (F[0,0,8,0], F[0,0,8,1]),               #chin
                            (F[0,0,36,0], F[0,0,36,1]),             #left eye corner 
                            (F[0,0,45,0], F[0,0,45,1]),             #right eye corner 
                            (F[0,0,48,0], F[0,0,48,1]),             #left mouth corner 
                            (F[0,0,54,0], F[0,0,54,1])],            #right mouth corner  
                            dtype="double")
    
 
        #camera matrix defaults/no camera distortion 
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], 
                         dtype = "double")


        #find pose estimate using cv solvePnP
        solved=[]
        solved = cv2.solvePnP(model_points_3d, facial_markers, camera_matrix, cam_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE) 

        #project the 3d points to image for 3 axis
        projected_nose_x=[]
        projected_nose_x=cv2.projectPoints(np.array([(600.0,0.0,0.0)]), solved[1], solved[2], camera_matrix, cam_dist_coeffs)
        projected_nose_y=[]
        projected_nose_y=cv2.projectPoints(np.array([(0.0,600.0,0.0)]), solved[1], solved[2], camera_matrix, cam_dist_coeffs)
        projected_nose_z=[]
        projected_nose_z=cv2.projectPoints(np.array([(0.0,0.0,600.0)]), solved[1], solved[2], camera_matrix, cam_dist_coeffs)


        #display the key facial markers
        for point in facial_markers:
            cv2.circle(im, (int(point[0]), int(point[1])), 3, (0,0,255), -1)

        #display head pose arrows
        np1 = ( int(facial_markers[0][0]), int(facial_markers[0][1]))
        np2 = ( int(projected_nose_x[0][0][0][0]), int(projected_nose_x[0][0][0][1]))
        np3 = ( int(projected_nose_y[0][0][0][0]), int(projected_nose_y[0][0][0][1]))
        np4 = ( int(projected_nose_z[0][0][0][0]), int(projected_nose_z[0][0][0][1]))
        cv2.arrowedLine(im, np1, np2, (255,0,0), 1)
        cv2.arrowedLine(im, np1, np3, (0,255,0), 1)
        cv2.arrowedLine(im, np1, np4, (0,0,255), 1)
    
    #display frame
    cv2.imshow('Output', im)

    #press q to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  else:
      break

#release capture
cap.release()

#close all frames
cv2.destroyAllWindows()
