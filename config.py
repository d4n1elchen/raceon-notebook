import yaml
import pickle
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
RESOLUTION = tuple(param['camera']['resolution'])
FRAMERATE = param['camera']['framerate']

# Vision params
CENTER_OFFSET = param['vision']['centerOffset']
SCAN_LINE = param['vision']['scaneLine']
PEAK_THRES = param['vision']['peakThres']
TRACK_WIDTH = param['vision']['trackWidth']

UNDISTORT_ENABLE = param['vision']['undistort']['enable']
K = np.array(param['vision']['undistort']['K'])
D = np.array(param['vision']['undistort']['D'])
PARAM_FILE = param['vision']['undistort']['paramFile']
with open(PARAM_FILE, "rb") as f:
    MAP1, MAP2 = pickle.load(f)

# Motor params
SPEED = param['motor']['speed']
SPEED_MIN = param['motor']['speedMin']

# Run timer
RUN_TIMER = param['run']['time']

# Log setting
LOG_FRAME = param['logging']['frame']
LOG_CYCLE_TIME = param['logging']['cycleTime']
LOG_SCAN_LINE = param['logging']['scanLine']

### Initialize components
# Camera
res = RESOLUTION
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
DECREASE_MAX = SPEED - SPEED_MIN

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
def findLinePos(I, line_pos=None, undistort_enable=False, scan_hist=None):
    # Undistort image
    if undistort_enable:
        I = undistort(I)
    
    # Select a horizontal line in the middle of the image
    L = I[SCAN_LINE, :]

    # Smooth the transitions so we can detect the peaks 
    Lf = filtfilt(b, a, L)
    
    # Log scan line
    if scan_hist != None:
        scan_hist.append(Lf)

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
        line_pos   = line_left + int(TRACK_WIDTH / 1)
        
    elif not line_left and line_right:
        line_pos   = line_right - int(TRACK_WIDTH / 1)
        
    else:
        if line_pos == None:
            line_pos = CAMERA_CENTER
        print(">>>> no line <<<<")
        print("Current pos = ", line_pos, ", peaks = ", peaks)
    
    return line_pos

# Undistor
DIM = RESOLUTION
def undistort(img, balance=0.0, dim2=None, dim3=None):
    undistorted_img = cv2.remap(img, MAP1, MAP2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

# Save image list
def saveFrameList(lst, di):
    for i, f in enumerate(lst):
        fig = plt.figure()
        if UNDISTORT_ENABLE:
            f = undistort(f)
        plt.imshow(f)
        plt.hlines(SCAN_LINE, 0, f.shape[1], 'w')
        fig.savefig(f"{di}/{i}.png")

# Plot histogram of cycle time
def plotCycleTime(t_hist, di, framerate, fname="cycle_time"):
    t = np.gradient(np.array(t_hist))
    fig = plt.figure()
    plt.hist(t, bins=10)
    plt.title(f"framerate = {framerate}, min_rate = {1/t.max():.2f}, max_rate = {1/t.min():.2f}")
    fig.savefig(f"{di}/{fname}.png")

print("Config has been loaded!")