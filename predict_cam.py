from keras.preprocessing import image
import numpy as np
# from keras.models import Model, load_model 
from mobilenet_v2 import MobileNetv2
import cv2
model = MobileNetv2((224,224, 3), 2)
model.load_weights("weights_train_colab2.h5")
label_dict=["glass","plastic"]
cap=cv2.VideoCapture(0)
while True:
    r,img_path= cap.read()
    img =cv2.resize(img_path,(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    x=x/255.0
    # print(x[0])

    preds = model.predict(x)
    preds=np.argmax(preds)
    print(preds)
    print(label_dict[int(preds)])
    cv2.imshow("Frame", img_path)
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cv2.release()
cv2.destroyAllWindows()
