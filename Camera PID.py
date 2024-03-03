# ME35 Rainbow Road Project Feb 2024
# Core opencv code provided by Einsteinium Studios
# Revisions to work with Pi Camera v3 by Briana Bouchard
# Revisions to operate robot by Vivian Becker and Jack Goldberg




import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import controls
import time
import RPi.GPIO as GPIO






#color path (SET THESE BEFORE START)
color_path = ['blue','black']
directions = ['stop']




curr_path_index = 0




# calibrate the gain constants
gainP = 0.175
gainD = 0.012
gainI = 0.002 #0.008




#motor variables


df_DC = 35 #25 # default duty cycle
frequency = 100
minSpeed = 15
maxSpeed =80




# ---------------------------------------------------------
GPIO.setmode(GPIO.BOARD)




picam2 = Picamera2() # assigns camera variable
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) # sets auto focus mode
picam2.start() # activates camera




time.sleep(1) # wait to give camera time to start up




# MOTOR PINS
motorL1 = 38
motorL2 = 36
motorR1 = 11
motorR2 = 13




GPIO.setup(motorL1, GPIO.OUT)
GPIO.setup(motorR1, GPIO.OUT)
GPIO.setup(motorL2, GPIO.OUT)
GPIO.setup(motorR2, GPIO.OUT)




GPIO.output(motorL2, GPIO.LOW)
GPIO.output(motorR2, GPIO.LOW)




pwm_L = GPIO.PWM(motorL1, frequency)  # channel=motorL frequency=50Hz
pwm_R = GPIO.PWM(motorR1, frequency)  # channel=motorR frequency=50Hz




pwm_L.start(0) # Starts the motor duty cycle at 0
pwm_R.start(0) # Starts the motor duty cycle at 0




# MOTOR CONTROL FUNCTIONS
def checkSpeed(speed):
  if speed<minSpeed:
      speed=minSpeed
  if speed>maxSpeed:
      speed=maxSpeed
  return speed
 speedL = df_DC
speedR = df_DC




# function to turn right
def motorControl(speedChange):
  
   #global speedL, speedR
   speedL = df_DC
   speedR = df_DC


   speedL = speedL - speedChange
   speedR = speedR + speedChange


   speedL = checkSpeed(speedL)
   speedR = checkSpeed(speedR)
  
   # if (speedL == speedR):
   #     print('forward')
   # elif (speedL > speedR):
   #     print('turn right')
   # elif (speedL < speedR):
   #     print('turn left')


   pwm_L.ChangeDutyCycle(speedL)
   pwm_R.ChangeDutyCycle(speedR)




# function to stop robot
def stop():
 pwm_R.ChangeDutyCycle(0)
 pwm_L.ChangeDutyCycle(0)




# function to calculate the error
def Error(cx):
  # if cx to the right of center, error is negative --> turn left
  # if cx to the left of center, error is positive --> turn right
  return center - cx




prev_error = 0
prev_time = time.time()
integral = 0




# function to calculate the derivative term
def IntegralandDerivative(error):
  global integral, prev_error, prev_time
  curr_time = time.time()
  dt = curr_time - prev_time


  # calculate integral term
  integral += error * dt


  # calculate derivative term
  # avoid divison by zero
  if dt == 0:
      return integral, 0
  # get instantaneous derivative
  d_error = (error - prev_error) / dt


  prev_error = error
  prev_time = curr_time


  return integral, d_error




try:




  while True:
    
       # Display camera input
       image = picam2.capture_array("main")
       cv2.imshow('img',image)
       # dimensions of cropped image
       xmin = 80
       xmax = 160*3+80
       ymin = 60
       ymax = 120*3




       # setpoint
       center = (xmax-xmin)/2


       # Crop the image
       crop_img = image[ymin:ymax, xmin:xmax]


       # Convert to grayscale
       gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
  
       # Gaussian blur
       blur = cv2.GaussianBlur(gray,(5,5),0)
  
       # Color thresholding
       # 80 < intensities < 255
       input_threshold,comp_threshold = cv2.threshold(blur,80,255,cv2.THRESH_BINARY_INV)
  
       # Find the contours of the frame
       contours,hierarchy = cv2.findContours(comp_threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  
      
       if len(contours) > 0:
          
           c = max(contours, key=cv2.contourArea) # Find the biggest contour (if detected)
           M = cv2.moments(c) # determine moment - weighted average of intensities (includes COM)




           # if zeroth moment (total area) is non zero (valid contour in image)
           if int(M['m00']) != 0:
               # m10: sum of the product of a pixel's intensity and its coordinate along the x axis --> CoM
               # m00: sum of intensities of all pixels inside contour
               # m10/m00: pixels more towards the center of mass of the contour contribute more to the average
               cx = int(M['m10']/M['m00']) # find x component of centroid location
               cy = int(M['m01']/M['m00']) # find y component of centroid location
           else:
               print("Centroid calculation error, looping to acquire new values")
               continue
           # line starts at cx and spans from top (0) to bottom (720)
           cv2.line(crop_img,(cx,0),(cx,720),(255,0,0),1) # display vertical line at x value of centroid
           cv2.line(crop_img,(0,cy),(1280,cy),(255,0,0),1) # display horizontal line at y value of centroid
  
           cv2.drawContours(crop_img, contours, -1, (0,255,0), 2) # display green lines for all contours
          
           # determine location of centroid in x direction and adjust steering recommendation
           error = Error(cx)
           integral, d_error = IntegralandDerivative(error)




           speedChange = gainP * error + gainI * integral + gainD * d_error
          
           print(speedChange)
          
           motorControl(speedChange)
  
       else:
           print("I don't see the line")
  
       # Display the resulting frame
       cv2.imshow('frame',crop_img)
      
       # Show image for 1 ms then continue to next image
       cv2.waitKey(1)




except KeyboardInterrupt:
  stop()
  print('All done')





