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

# Logging
if LOG_FRAME:
    frame_hist = []
if LOG_CYCLE_TIME:
    t_hist = []
if LOG_SCAN_LINE:
    scan_hist = []

# Loop starting time
t_now = time.time()
if LOG_CYCLE_TIME:
    t_hist.append(t_now)

line_pos = None
for f in stream:
    if first_frame:
        first_frame = False
        # Reset the buffer for the next image
        rawCapture.truncate(0)
        # Save first frame
        if LOG_FRAME:
            frame_hist.append(f.array)
            frame_hist.append(f.rgb_array)
        continue
    
    # Stop after RUN_TIMER seconds
    if (t_now - t) > RUN_TIMER:
        # Save last frame
        if LOG_FRAME:
            frame_hist.append(f.array)
            frame_hist.append(f.rgb_array)
        break
    
    # Get the intensity component of the image (a trick to get black and white images)
    I = f.array[:, :, 0]
    
    # Reset the buffer for the next image
    rawCapture.truncate(0)
    
    if LOG_SCAN_LINE:
        line_pos = findLinePos(I, line_pos, undistort_enable=UNDISTORT_ENABLE, scan_hist=scan_hist)
    else:
        line_pos = findLinePos(I, line_pos, undistort_enable=UNDISTORT_ENABLE)
    
    # Update PID
    pid.update(line_pos)
    DUTY_CYCLE = servoLimit(SERVO_MIDDLE + int(pid.output)) 
    
    # Update servo and motor
    servo.duty_cycle = DUTY_CYCLE
    motor.duty_cycle = (MOTOR_BRAKE + SPEED) - decreaseSpeedPos(line_pos)
    
    # Time logging
    t_now = time.time()
    if LOG_CYCLE_TIME:
        t_hist.append(t_now)

### Endloop ###

# Stop the car and reset server
motor.duty_cycle = MOTOR_BRAKE   
servo.duty_cycle = SERVO_MIDDLE 

# Release resources
stream.close()
rawCapture.close()
camera.close()

# Post process
if LOG_FRAME:
    saveFrameList(frame_hist, "photo")
if LOG_CYCLE_TIME:
    plotCycleTime(t_hist, "plot", FRAMERATE, fname="cycle_time")
if LOG_SCAN_LINE:
    np.save('scan_hist.npy', np.array(scan_hist))