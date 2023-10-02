""""
Developer or author : S.Vigneswaran

Date                : 01.10.2023

***Hello Developers***

I developed this program to find How many times you are blink your eyes for a time period(Until you close the program)

This program use Mediapipe to detect face mesh.

The face mesh have 460+ landmarks(it will vary in new releases).

I sorted the landmarks to find the eyelid

I hope you will learn many concepts by my program

"""

import os

try:
    import cv2              # We import cv2(opencv-python) to access your camera and handle graphic related works
    import mediapipe as mp  # This library or frame work developed by google to do some ML works ans use pre-trained model


except:                     # If we get any error in import it will reinstall or if stuck on no module error it will install newly from Pipy
    
    print("can't find necessary modules . wait for a minute to install")

    os.system("pip install opencv-python") 
    os.system("pip install mediapipe")

import math # It is used to do some math calcution
import time # It is used to do time related works

blinks = 0     # We initialize blink as 0
closed = False # We set closed(eye closed) as Flse

instruction_disclaimer = "This is created by Vigneswaran\nIf you want to close stop this program press 'Esc'\nDisclaimer: It will not perfect output will have plus or minus 5 blinks Tolarance"

mp_drawing = mp.solutions.drawing_utils # It used to make anntations and draw line between landmarks
mp_face_mesh = mp.solutions.face_mesh   # It used to find face mesh from your face

cap = cv2.VideoCapture(0)               # We set the defualt camera as video capturing device

def find_distance(point_one,point_two):

    """"This function used to find distance between two points by there x,y values
        
        We use https://www.onlinemath4all.com/images/pythagoreantheorem24.png formula to find distance between to points """

    point1 = (face_landmarks.landmark[point_one].x,face_landmarks.landmark[point_one].y) #Find the x and y of point one
    point2 = (face_landmarks.landmark[point_two].x,face_landmarks.landmark[point_two].y) #Find the x and y od point two
    distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)        #We convert that formula to like this to use in python
    distance = round(distance,3)                                                         #the distance variable is like Hanuman's tail . So we need to round that to 3 digits
    return distance*10                                                                   #Mostly the the distance will be less than 1 so we need to mutiply by 10 

with mp_face_mesh.FaceMesh(         # This line initializes the FaceMesh model with specified parameters. It sets up the FaceMesh model to detect facial landmarks on a single face with minimum detection and tracking confidence scores of 0.5.
    max_num_faces=1,                # Limits the model to detect only one face at a time. if we choose multiple face the time blinks will be wrong
    refine_landmarks=True,          # Indicates that the model should refine facial landmarks for improved accuracy.
    min_detection_confidence=0.5,   # Sets the minimum confidence threshold for face detection to 0.5 (50% confidence).
    min_tracking_confidence=0.5     # Sets the minimum confidence threshold for landmark tracking to 0.5 (50% confidence).
    ) as face_mesh:

    starting = time.time()          # We use this to find the initialized time of the program
    
    while cap.isOpened():          # This line starts a while loop that will continuously run as long as the cap object (camera capture) is opened. It forms the main loop of your program where video frames are processed.
        
        
        success, image = cap.read()  # Inside the loop, this line captures a frame from the camera using cap.read(). It returns two values: success (a boolean indicating whether the frame was successfully captured) and image (the captured frame).
        

        # Here, the code checks if the frame capture was successful (success is False when there's no frame to capture). If unsuccessful, it prints a message and continues to the next iteration of the loop.
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # These lines temporarily make the image non-writeable to ensure compatibility with MediaPipe. Then, it converts the image from the BGR color space to RGB color space, which is the format MediaPipe expects.
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # This line uses the face_mesh model to process the RGB image and obtain results, which include detected facial landmarks.
        results = face_mesh.process(image_rgb)

        # After obtaining the results, the code makes the image writeable again and converts it back to the BGR color space for further processing and displaying.
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


        # This conditional checks if any facial landmarks were detected in the current frame. It enters this block if at least one face is detected.Inside the loop, the code iterates through each detected face's landmarks. You will perform subsequent calculations and drawing on each face individually.
        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:
               

                mouth_landmarks = [159,23] # these are the landmark for our eye and eyelid

                eye_d = find_distance(159,23) # We find the distance between that two points
                
            
                # If the we blink our eyes that distance between the eyelids will decreased
                # If the distance is decreased to less than 0.25 that is the sign of closed eyes 
                if(eye_d < 0.25):

                    # only add 1 if the eye is closed because if the user closes for some seconds . It will add continuously and one 
                    if(closed == False):
                        closed = True  
                        blinks += 1 #Add plus one blink
                    
                else:
                    closed = False # Else the distance is more than 0.25 it will be false


                # Line number 117 to 128 used to put user guidence text on the window

                y_position = 17

                for line in instruction_disclaimer.split("\n"):

                    if(y_position == 54):

                        cv2.putText(image, line, (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 25), 2, cv2.LINE_AA)


                    else:
                        cv2.putText(image, line, (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 25), 1, cv2.LINE_AA)

                    y_position += 22  

                cv2.putText(image,f"Total blinks {blinks}",(60,125),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,250),1,cv2.LINE_AA) # Update the current blink values

        cv2.imshow('MediaPipe Mouth Mesh',image) # Finally show our image
    
        if cv2.waitKey(5) & 0xFF == 27:          # if user press Esc button on their key board the program will be stoped
            break
  
cap.release()             # Releases the video capture resources
cv2.destroyAllWindows()   # Closes all OpenCV windows, ensuring proper cleanup after video processing.

end = time.time()         # Find the ending time
times = (end-starting)/60 # We find how long the the program was runned by ' end-staring ' it will give in seconds . But minutes is comfortable so we divide by 60.

print(f"you are blinked {str(blinks)} times in {str(round(times,2))} minutes") # print the results  