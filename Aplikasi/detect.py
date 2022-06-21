import tensorflow as tf
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
cv2.namedWindow("Classification", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Classification', 640, 640) 

while True:
    ret, frame = camera.read()
    print("[INFO] loading and preprocessing image...")
    frame = cv2.resize(frame, (200,200), interpolation=cv2.INTER_AREA)
    image = tf.convert_to_tensor(frame, dtype=tf.float32)
    image = np.expand_dims(image, axis=0)

    print("[INFO] loading network...")
    MODEL_PATH = 'model/model_aug.h5'

    # Load your trained model
    model = load_model(MODEL_PATH)

    #classify the image
    print("[INFO] classifying image...")
    preds = model.predict(image)
    classes = ['Mata Tertutup', 'Mata Terbuka', 'Tidak Menguap', 'Menguap']  
    predict = classes[np.argmax(preds, axis=1)[0]] 

    print("Label: {}".format(predict))
    cv2.putText(frame, "Label: {}".format(predict), (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imshow("Classification", frame)
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()    