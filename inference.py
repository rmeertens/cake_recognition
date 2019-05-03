import cv2
import sys
import tensorflow as tf
import PIL
import picamera
import picamera.array

import numpy as np

model = tf.keras.models.load_model('cake_recognition_model.h5')

with tf.Session(graph=inp.graph):
    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            camera.resolution = (640, 480)

            while True:
                camera.capture(stream, 'rgb', use_video_port=True)
                stream.flush()
                # stream.array now contains the image data in BGR order
                new_im = PIL.Image.fromarray(stream.array)

                # img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
                # x = tf.keras.preprocessing.image.img_to_array(new_im)
                # assert(x.shape == (224, 224, 3))
                #
                # images.append(x)
                # labels.append(label)
                # temp = np.array(images)
                # unresized_images = tf.keras.applications.mobilenet.preprocess_input(temp)
                #
                # inference_on= np.array(unresized_images)
                #
                #
                # img = np.array(new_im.resize((96,96))).astype(np.float) / 128 - 1
                cv2.imshow('image',stream.array)
                cv2.waitKey(10)
                # x = predictions.eval(feed_dict={inp: img.reshape(1, 96,96, 3)})
                # print(x.argmax())
                # print(imagenet_labels[x.argmax()-1])
                # # reset the stream before the next capture
                stream.seek(0)
                stream.truncate()

            cv2.destroyAllWindows()
