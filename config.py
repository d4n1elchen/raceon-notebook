import yaml
import numpy as np
from picamera.array import PiYUVArray, PiRGBArray
from picamera import PiCamera
from pwm import PWM
from pprint import pprint
from scipy.signal import find_peaks, butter, filtfilt

print("Load config ...")

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

with open("parameters.yaml", 'r') as ymlfile:
    param = yaml.load(ymlfile, Loader=Loader)

pprint(param)

### Load parameters
# PID
Kp = param['pid']['kp']
Ki = param['pid']['ki']
Kd = param['pid']['kd']

# Camera settings
FRAMERATE = param['camera']['framerate']

# Vision params
CENTER_OFFSET = param['vision']['centerOffset']
SCAN_LINE = param['vision']['scaneLine']
PEAK_THRES = param['vision']['peakThres']
TRACK_WIDTH = param['vision']['trackWidth']

# Motor params
SPEED = param['motor']['speed']

# Run timer
RUN_TIMER = param['run']['time']


### Initialize components
# Camera
res = (640, 480)
camera = PiCamera()
camera.sensor_mode = 7
camera.resolution = res
camera.framerate = FRAMERATE
rawCapture = PiYUVArray(camera, size=res)
stream = camera.capture_continuous(rawCapture, format="yuv", use_video_port=True)

# Enable servo
SERVO_MIDDLE = 1500000
SERVO_MAX = 2000000
SERVO_MIN = 1000000

servo = PWM(1)
servo.period = 20000000
servo.duty_cycle = SERVO_MIDDLE
servo.enable = True

# Enable motor
MOTOR_BRAKE = 1000000
DECREASE_MAX = SPEED - 100000

motor = PWM(0)
motor.period = 20000000
motor.duty_cycle = MOTOR_BRAKE
motor.enable = True

# Vision
CAMERA_CENTER = res[0] // 2 + CENTER_OFFSET
# To filter the noise in the image we use a 3rd order Butterworth filter
# Wn = 0.02, the cut-off frequency, acceptable values are from 0 to 1
b, a = butter(3, 0.1)

### Helping function
# Servo limit
def servoLimit(DUTY_CYCLE):
    if DUTY_CYCLE > SERVO_MAX:
        return SERVO_MAX
    if DUTY_CYCLE < SERVO_MIN:
        return SERVO_MIN
    return DUTY_CYCLE

# Decrease speed when turning
def decreaseSpeedPos(line_pos):
    return int(DECREASE_MAX * np.tanh(abs(line_pos) / CAMERA_CENTER))
def decreaseSpeedDuty(DUTY_CYCLE):
    return int(DECREASE_MAX * abs(DUTY_CYCLE - SERVO_MIDDLE) / SERVO_MIDDLE)

# Get line_pos
def findLinePos(I):
    # Select a horizontal line in the middle of the image
    L = I[SCAN_LINE, :]

    # Smooth the transitions so we can detect the peaks 
    Lf = filtfilt(b, a, L)

    # Find peaks which are higher than 0.5
    p = find_peaks(Lf, height=PEAK_THRES)
    
    peaks = p[0]
    
    line_left   = None
    line_right  = None
    peaks_left  = peaks[peaks < CAMERA_CENTER]
    peaks_right = peaks[peaks > CAMERA_CENTER]

    if peaks_left.size:
        line_left = peaks_left[-1]

    if peaks_right.size:
        line_right = peaks_right[0]
        
    if line_left and line_right:
        line_pos   = (line_left + line_right) // 2
        
    elif line_left and not line_right:
        line_pos   = line_left + TRACK_WIDTH // 2
        
    elif not line_left and line_right:
        line_pos   = line_right - TRACK_WIDTH // 2
        
    else:
        line_pos = CAMERA_CENTER
        print(">>>> no line <<<<")
        print("Current pos = ", line_pos, ", peaks = ", peaks)
    
    line_pos = line_pos - CAMERA_CENTER
    
    return line_pos

print("Config has been loaded!")