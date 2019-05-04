import cv2
import sys
import tensorflow as tf
import PIL
import picamera
import picamera.array
import skimage

import numpy as np

model = tf.keras.models.load_model('improv_cake_recognition_model.h5')
#model = tf.contrib.saved_model.load_keras_model('food_model/a')
print("managed to load model")
with open('class_names.txt') as f:
    class_names = f.readline().split(',')[:-1]
print(class_names)

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as stream:
        camera.resolution = (640, 480)

        while True:
            camera.capture(stream, 'rgb', use_video_port=True)
            stream.flush()
            # stream.array now contains the image data in BGR order
           # new_im = PIL.Image.fromarray(stream.array)
            resized = skimage.transform.resize(stream.array, (224,224,3), preserve_range=True)  

            processed = tf.keras.applications.mobilenet.preprocess_input(resized)
            result = model.predict(np.array([processed]))
            print(class_names[np.argmax(result[0])])
            #
            #
            # img = np.array(new_im.resize((96,96))).astype(np.float) / 128 - 1
            #cv2.imshow('image',stream.array)
            #cv2.waitKey(10)
            # x = predictions.eval(feed_dict={inp: img.reshape(1, 96,96, 3)})
            # print(x.argmax())
            # print(imagenet_labels[x.argmax()-1])
            # # reset the stream before the next capture
            stream.seek(0)
            stream.truncate()

        cv2.destroyAllWindows()
