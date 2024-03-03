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
import math




#color path (SET THESE BEFORE START)
color_path = ['red', 'blue', 'red', 'black']
directions = ['R', 'R', 'end']




curr_path_index = 0




# calibrate the gain constants
gainP = 0.175
gainD = 0.012
gainI = 0.002


gainS = .7 # gain for the slope error


#motor variables


df_DC = 35 #25 # default duty cycle
frequency = 100
minSpeed = 15
maxSpeed =80


# area variables


max_contour_area = 85000 # area when 2 lines must be present
min_contour_area = 20000 # area when a line is present
min_half_contour_area = 5000 # area when some of the new line is present


# threshhold variables


# blue values
B_threshMin = 83
B_threshMax = 84


# green values
G_threshMin = 113
G_threshMax = 114


# red values
R_threshMin = 49
R_threshMax = 50


# purple values
P_threshMin = 0
P_threshMax = 0


# black values
Bl_threshMin = 40
Bl_threshMax = 41




# ---------------------------------------------------------
GPIO.setmode(GPIO.BOARD)




picam2 = Picamera2() # assigns camera variable
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) # sets auto focus mode
picam2.start() # activates camera




time.sleep(10) # wait to give camera time to start up




# # MOTOR PINS
# motorR1 = 38
# motorR2 = 36
# motorL1 = 11
# motorL2 = 13


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




pwm_L = GPIO.PWM(motorL1, frequency)
pwm_R = GPIO.PWM(motorR1, frequency)




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


#function to move forward a little bit
def forward():
   pwm_R.ChangeDutyCycle(df_DC)
   pwm_L.ChangeDutyCycle(df_DC)
   time.sleep(0.15)
   stop()
   time.sleep(.2)


# function to turn right at a given speed
def turnRight(turnSpeed):
   stop()
   time.sleep(.2)
   # turn right
   pwm_R.ChangeDutyCycle(0)
   pwm_L.ChangeDutyCycle(turnSpeed)
   time.sleep(.5)
   stop()


# function to turn left at a given speed
def turnLeft(turnSpeed):
   stop()
   time.sleep(.2)
   # turning left
   pwm_R.ChangeDutyCycle(turnSpeed)
   pwm_L.ChangeDutyCycle(0)
   time.sleep(.5)
   stop()




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


# function to find intersection
def processIntersection(blur, minThresh, maxThresh):
  
   # Color thresholding
   input_threshold,comp_threshold = cv2.threshold(blur,minThresh,maxThresh,cv2.THRESH_BINARY_INV)


   # Find the contours of the frame
   contours,hierarchy = cv2.findContours(comp_threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  
   C = max(contours, key=cv2.contourArea) # Find the biggest contour (if detected)


   M = cv2.moments(C) # determine moment - weighted average of intensities (includes COM)


   slope = get_slope(C)


   if int(M['m00']) > min_contour_area:
       return True, slope
   else:
       return False, 0


# function to find centroid
def findCentroid(M):
   # m10: sum of the product of a pixel's intensity and its coordinate along the x axis --> CoM
   # m00: sum of intensities of all pixels inside contour
   # m10/m00: pixels more towards the center of mass of the contour contribute more to the average
   cx = int(M['m10']/M['m00']) # find x component of centroid location
   cy = int(M['m01']/M['m00']) # find y component of centroid location
   return cx,cy


# function to draw horizontal and vertical lines at the centroid
def draw(image, cx, cy):
   # line starts at cx and spans from top (0) to bottom (720)
   cv2.line(image,(cx,0),(cx,720),(255,0,0),1) # display vertical line at x value of centroid
   cv2.line(image,(0,cy),(1280,cy),(255,0,0),1) # display horizontal line at y value of centroid


   cv2.drawContours(image, contours, -1, (0,255,0), 2) # display green lines for all contours


   return image


# get slope from the point distribution of the new contour
def get_slope(contour):


   # Compute the covariance matrix of the contour
   mean, eigenvectors = cv2.PCACompute(contour, mean=None)


   # Get the principal component (major axis)
   major_axis = eigenvectors[0]


   # Calculate the slope using the major axis
   slope = major_axis[1] / major_axis[0]


   return slope


# function to rotate until vertical line is the same as slope
def followNewLine (slope):
   error_angle = abs((math.pi-math.atan(slope)))


   change = error_angle*gainS


   turnSpeed = df_DC + change


   if directions[curr_path_index] == 'R':
       turnRight(turnSpeed)
   elif directions[curr_path_index] == 'L':
       turnLeft(turnSpeed)
   elif directions[curr_path_index] == 'end':
       stop()
       return 'endFound'
   return 'intPassed'


end = False


try:


   time.sleep(.1)


   while not end:
    
       intFound = False
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
       #crop_img = image




       # Convert to grayscale
       gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
  
       # Gaussian blur
       blur = cv2.GaussianBlur(gray,(5,5),0)
  
       # Color thresholding
       # 80 < intensities < 255
       input_threshold,comp_threshold = cv2.threshold(blur,100,255,cv2.THRESH_BINARY_INV)
  
       # Find the contours of the frame
       contours,hierarchy = cv2.findContours(comp_threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  
       # if zeroth moment (total area) is non zero (valid contour in image)
       if len(contours) > 0:
          
           c = max(contours, key=cv2.contourArea) # Find the biggest contour (if detected)
           M = cv2.moments(c) # determine moment - weighted average of intensities (includes COM)


          
           # if area is greater than a max amount for only one tape present, look for intersection in og blurred image
           if int(M['m00']) > max_contour_area:
               # only look for the intersections that we care about
               if color_path[curr_path_index+1] == 'blue':
                   next_threshMin = B_threshMin
                   next_threshMax = B_threshMax
               elif color_path[curr_path_index+1] == 'green':
                   next_threshMin = G_threshMin
                   next_threshMax = G_threshMax
               elif color_path[curr_path_index+1] == 'red':
                   next_threshMin = R_threshMin
                   next_threshMax = R_threshMax
               elif color_path[curr_path_index+1] == 'purple':
                   next_threshMin = P_threshMin
                   next_threshMax = P_threshMax
               elif color_path[curr_path_index+1] == 'black':
                   next_threshMin = Bl_threshMin
                   next_threshMax = Bl_threshMax


               intersectionFound = False


               # find next tape color in blurred image and make a contour
               intersectionFound, slope = processIntersection(blur, next_threshMin, next_threshMax)
              
               if intersectionFound == True:
                   # correct intersection found


                   result = followNewLine(slope)
                  
                   if result == 'endFound':
                       end = True
                   if result == 'intPassed':
                       intFound = True
                       forward()
                       curr_path_index += 1
               else:
                   # correct intersection not found
                   pass


           elif int(M['m00']) <= max_contour_area:
               cx,cy = findCentroid(M)
           else:
               print("Centroid calculation error, looping to acquire new values")
               continue
          
           if not intFound:
               cx,cy = findCentroid(M)


               crop_image = draw(crop_img, cx, cy)
                          
               error = Error(cx)
               integral, d_error = IntegralandDerivative(error)


               speedChange = gainP * error + gainI * integral + gainD * d_error
                          
               motorControl(speedChange)
           else:
               pass
  
       else:
           print("I don't see the line")
  
       # Display the resulting frame
       cv2.imshow('frame',crop_img)
      
       # Show image for 1 ms then continue to next image
       cv2.waitKey(1)




except KeyboardInterrupt:
  stop()
  print('All done')





