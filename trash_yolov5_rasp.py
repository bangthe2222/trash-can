import numpy as np
import cv2
import time
from time import sleep
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import time_synchronized
from gpiozero import DistanceSensor,Button
import RPi.GPIO as GPIO
import time
# setup GPI0
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
# setup yolov5
model = attempt_load("best.pt", map_location="cpu")  # load FP32 model
names = model.module.names if hasattr(model, 'module') else model.names
def detect(img):
    img = torch.from_numpy(img).to("cpu")
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t1 = time_synchronized()
    pred = model(img)[0]
    pred = non_max_suppression(pred,float(0.4), float(0.45))
    print(pred)
    t2 = time_synchronized()
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s =''
        s += '%gx%g ' % img.shape[2:]  # print string
        if len(det):
            for c in det[:, -1].unique():
                n_bottle=0
                name=names[int(c)]
                n = (det[:, -1] == c).sum() 
                if name=="plastic":
                    n_bottle=f"{n}"
                 # detections per class
                s += f"{n} {names[int(c)]} {'' * (n > 1)}, "  # add to string
        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')
    return n_bottle 
# yolov5 convert
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
# raspberry Pi4 device
def servo1():
	GPIO.setup(14, GPIO.OUT)
	pwm = GPIO.PWM(14, 100)
	pwm.start(0)
	duty1 = float(90) / 10.0 + 2.5
	pwm.ChangeDutyCycle(duty1)
	time.sleep(0.5)
	duty1 = float(180) / 10.0 + 2.5
	pwm.ChangeDutyCycle(duty1)
	time.sleep(0.5)
	GPIO.cleanup(18)
def servo2():
	GPIO.setup(15, GPIO.OUT)
	pwm = GPIO.PWM(15, 100)
	pwm.start(0)
	duty1 = float(180) / 10.0 + 2.5
	pwm.ChangeDutyCycle(duty1)
	time.sleep(0.5)
	duty1 = float(90) / 10.0 + 2.5
	pwm.ChangeDutyCycle(duty1)
	time.sleep(0.5)
	GPIO.cleanup(18)
def servo3(angle):
	GPIO.setup(18, GPIO.OUT)
	pwm = GPIO.PWM(18, 100)
	pwm.start(0)
	duty1 = float(angle) / 10.0 + 2.5
	pwm.ChangeDutyCycle(duty1)
	time.sleep(0.5)
	duty1 = float(90) / 10.0 + 2.5
	pwm.ChangeDutyCycle(duty1)
	time.sleep(0.5)
	GPIO.cleanup(18)
def distance():
	sensor = DistanceSensor(echo=24, trigger=23)
	return sensor.distance*100
def button():
	button=Button(25)
	mode=""
	if button.is_pressed:
		mode="On"
		print("pressed")
	return mode
# load our serialized model from disk
print("[INFO] loading model...")

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
cap=cv2.VideoCapture(0)
score=0
sleep(3)
# loop over the frames from the video stream
while cv2.waitKey(1)!=30:
	# resize the video stream window at a maximum width of 500 pixels
	distance()

	if distance()<=10:
		time.sleep(0.5)
		label="None"
		t1=time.time()
		r,frame = cap.read()
		frame =cv2.flip(frame,-1)
		img = letterbox(frame, 320, stride=32)[0]
		img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
		img = np.ascontiguousarray(img)
		n_bottle=detect(img)
		if n_bottle!=0:
			servo3(180)
			score+=n_bottle
			print(score)
		else:
			servo3(10)
		t2=time.time()
		if t2-t1>=20:
			score=0
	button1=button()
	if (button1=="On") and (10>score>=5):
		servo1()
		score=0
	if (button1=="On") and (score>=10):
		servo2()
		score=0
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
# cleanup
cv2.release()
cv2.destroyAllWindows()

