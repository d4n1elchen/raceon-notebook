import yaml
import pickle
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
Kp_straight = param['pid']['straight']['kp']
Ki_straight = param['pid']['straight']['ki']
Kd_straight = param['pid']['straight']['kd']
Kp_turn = param['pid']['turn']['kp']
Ki_turn = param['pid']['turn']['ki']
Kd_turn = param['pid']['turn']['kd']
USE_STATE = param['pid']['state']['enable']
SAVE_STATE = param['pid']['state']['save']
STATE_FILE = param['pid']['state']['file']
if USE_STATE:
    with open(STATE_FILE, "rb") as f:
        PID_STATE = pickle.load(f)

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
LOG_LINE_POS = param['logging']['linePos']


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
    return int(DECREASE_MAX * decreaseCurve((line_pos - CAMERA_CENTER) / CAMERA_CENTER))
def decreaseSpeedDuty(DUTY_CYCLE):
    return int(DECREASE_MAX * decreaseCurve((DUTY_CYCLE - SERVO_MIDDLE) / SERVO_MIDDLE))
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def decreaseCurve(x):
    #return np.maximum(0.0, np.tanh((abs(x)-0.05)*16))
    return sigmoid((abs(x)-0.07)*203)

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
    peaks, p_val = find_peaks(Lf, height=PEAK_THRES)
    
    line_left   = None
    line_right  = None
    peaks_left  = peaks[peaks < line_pos]
    peaks_right = peaks[peaks > line_pos]
    
    state = 0

    if peaks_left.size:
        line_left = peaks_left[-1]

    if peaks_right.size:
        line_right = peaks_right[0]
        
    if line_left and line_right:
        print(">>>> 2 lines <<<<")
        line_pos   = (line_left + line_right) // 2
        print("Current pos = {:f}, Left = {:f}, Right = {:f}".format(line_pos, line_left, line_right))
        
    elif line_left and not line_right:
        print(">>>> only left <<<<")
        line_pos   = line_left + int(TRACK_WIDTH / 2)
        print("Current pos = {:f}, Left = {:f}".format(line_pos, line_left))
        state = 1
        
    elif not line_left and line_right:
        print(">>>> only right <<<<")
        line_pos   = line_right - int(TRACK_WIDTH / 2)
        print("Current pos = {:f}, Right = {:f}".format(line_pos, line_right))
        state = 1
        
    else:
        print(">>>> no line <<<<")
        print("Current pos = {:f}, peaks = {:s}, p_val = {:s}".format(line_pos, str(peaks), str(p_val['peak_heights'])))
    
    return line_pos, state

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

# Plot histogram of cycle time
def plotPos(pos_hist, di, fname="line_pos"):
    pos_hist = (np.array(pos_hist) - CAMERA_CENTER) / CAMERA_CENTER
    print("=== (Line pos - CAMERA_CENTER) / CAMERA_CENTER ===")
    print("Avg = {:.2f}, Min = {:.2f}, Max = {:.2f}".format(pos_hist.mean(), pos_hist.min(), pos_hist.max()))
    fig, ax1 = plt.subplots()
    ax1.plot(pos_hist)
    ax1.set_title("Line Position")
    ax1.set_ylim(-0.2, 0.2)
    ax2 = ax1.twinx()
    ax2.plot(decreaseCurve(pos_hist), color='r')
    fig.savefig(f"{di}/{fname}.png")

print("Config has been loaded!")