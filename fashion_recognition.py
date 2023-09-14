import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('fashion_mnist.h5')
items = ['Tshirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the image
img = cv2.imread(r"Image path", 0)  # read and convert the image to grayscale
img = cv2.resize(img, (28, 28))  # resize the image to 28x28 pixels
img = np.invert(np.array([img]))  # invert the image
prediction = model.predict(img)
print("The item is probably a {}".format(items[np.argmax(prediction)]))
# Display the image
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
