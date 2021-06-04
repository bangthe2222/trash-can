import numpy as np
import cv2
import time
from time import sleep
from gpiozero import DistanceSensor,Button
import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

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

prototxtPath =r"./realtime-object-detection-master/MobileNetSSD_deploy.prototxt.txt"
weightsPath = r"./realtime-object-detection-master/MobileNetSSD_deploy.caffemodel"
# initialize the list of class labels MobileNet SSD was trained to detect
# and generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size = (len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNet(prototxtPath,weightsPath)

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
		label="None"
		t1=time.time()
		r,frame = cap.read()
		frame =cv2.flip(frame,-1)
		# frame = imutils.resize(frame, width=500)
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
		# pass the blob through the network and get the detections
		net.setInput(blob)
		detections = net.forward()
		# loop over the detections
		list=[]
		for i in np.arange(0, detections.shape[2]):
			# extract the probability of the prediction
			probability = detections[0, 0, i, 2]
			# filter out weak detections by ensuring that probability is
			# greater than the min probability
			if probability > 0.8:
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				# draw the prediction on the frame
				label = "{}".format(CLASSES[idx])
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				print(startX,endX,startY,endY)
				print(label)
				list.append(label)
		if "bottle" in list:
			servo3(180)
			score+=1
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

