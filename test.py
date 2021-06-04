# To run the script, execute the following commands
# workon cv
# python3 real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
# from imutils.video import FPS
import time
from keras.preprocessing import image
import numpy as np
# from keras.models import Model, load_model 
from mobilenet_v2 import MobileNetv2
import cv2
model = MobileNetv2((480,480, 3), 3)
model.load_weights("weights_train_colab.h5")
label_dict=["glass","other","plastic"]
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
cap=cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)
# fps = FPS().dem()
while True:
	# resize the video stream window at a maximum width of 500 pixels
	r,frame = cap.read()
	frame =cv2.flip(frame,1)
	# frame = imutils.resize(frame, width=500)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	# pass the blob through the network and get the detections
	net.setInput(blob)
	detections = net.forward()
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):
		global label
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
			if label=="bottle":
				imgs=frame[startY:endX,startX:endX]
				img =cv2.resize(imgs,(480,480))
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)
				# x = preprocess_input(x)
				x=x/255.0
				preds = model.predict(x)
				print(preds)
				preds=np.argmax(preds)
				print(preds)
				print(label_dict[int(preds)])
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
# cleanup
cv2.release()
cv2.destroyAllWindows()

