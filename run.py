import time
import PID

# Initialize all global variables
from config import *

input("Press Enter to continue ...")

# Measure the time needed to process 300 images to estimate the FPS
ti = t = time.time()

pid = PID.PID(Kp, Ki, Kd)
pid.SetPoint = 0
#pid.setSampleTime(1/camera.framerate)

# start car
motor.duty_cycle = MOTOR_BRAKE + SPEED

# Initialize variables
line_pos    = CAMERA_CENTER
first_frame = True

for f in stream:
    if first_frame:
        first_frame = False
        # Reset the buffer for the next image
        rawCapture.truncate(0)
        continue
    
    # Stop after RUN_TIMER seconds
    if (time.time() - t) > RUN_TIMER:
        break
    
    # Get the intensity component of the image (a trick to get black and white images)
    I = f.array[:, :, 0]
    
    # Reset the buffer for the next image
    rawCapture.truncate(0)
    
    line_pos = findLinePos(I)
    
    # Update PID
    pid.update(line_pos)
    DUTY_CYCLE = servoLimit(SERVO_MIDDLE + int(pid.output)) 
        
    servo.duty_cycle = DUTY_CYCLE
    motor.duty_cycle = (MOTOR_BRAKE + SPEED) - decreaseSpeedPos(line_pos)

### Endloop ###

# Stop the car and reset server
motor.duty_cycle = MOTOR_BRAKE   
servo.duty_cycle = SERVO_MIDDLE 

# Release resources
stream.close()
rawCapture.close()
camera.close()
