import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Load the saved model
model = tf.keras.models.load_model('mnist.h5')

#Load the image
img = cv2.imread(r"Image path", 0)# convert the image to grayscale
img = cv2.resize(img, (28, 28))# resize the image to 28x28 pixels
img = np.invert(np.array([img]))# invert the image
prediction = model.predict(img)
print("The number is probably a {}".format(np.argmax(prediction)))
#Display the image
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
