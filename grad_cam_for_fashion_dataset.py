import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('fashion_mnist.h5')

# Load and preprocess the image
image_path = r"Image path"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
img = cv2.resize(img, (28, 28))  # Resize to match model input size
img = img / 255.0  # Normalize to the range [0, 1]
img = np.expand_dims(img, axis=0)  # Add batch dimension
img = tf.convert_to_tensor(img, dtype=tf.float32)  # Convert to a TensorFlow tensor

# Define a function to generate a class activation map for a specific layer
def generate_cam(model, img_tensor, layer_index):
    # Define a new model that outputs the specified layer's activations
    cam_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[layer_index].output)

    # Get the model's prediction for the input image
    preds = model.predict(img_tensor)
    predicted_class = np.argmax(preds)

    # Get the output feature map of the specified layer
    with tf.GradientTape() as tape:
        activations = cam_model(img_tensor)
        class_output = activations[:, predicted_class]

    # Check if the layer produces CAM-compatible activations
    if activations.shape.ndims == 4 and activations.shape[-1] > 1:
        # Calculate the gradient of the class output with respect to the feature map
        grads = tape.gradient(class_output, activations)

        # Compute the global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each feature map by its corresponding gradient importance
        cam = tf.reduce_sum(tf.multiply(pooled_grads, activations), axis=-1).numpy()

        # Normalize CAM for visualization
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Resize CAM to match the original image size
        cam = cv2.resize(cam[0], (img_tensor.shape[2], img_tensor.shape[1]))
    else:
        # If the layer doesn't produce CAM-compatible activations, return a placeholder CAM
        cam = np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.float32)

    return cam

# Get the names of layers in the model
layer_names = [layer.name for layer in model.layers]

# Create a subplot grid
n_cols = 4
n_rows = len(layer_names) // n_cols + 1
fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 16))

# Generate CAMs for each layer
for i, layer_name in enumerate(layer_names):
    cam = generate_cam(model, img, i)

    # Apply a colormap to the CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_RAINBOW)

    # Ensure that both img_uint8 and heatmap have the same dimensions and data type
    img_uint8 = cv2.cvtColor(np.uint8(255 * img[0]), cv2.COLOR_GRAY2BGR)
    heatmap = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))

    # Overlay the heatmap on the original image
    overlaid_img = cv2.addWeighted(img_uint8, 0.5, heatmap, 0.5, 0)

    row = i // n_cols
    col = i % n_cols

    axs[row, col].imshow(overlaid_img)
    axs[row, col].set_title(layer_name)
    axs[row, col].axis('off')

# Remove empty subplots
for i in range(len(layer_names), n_cols * n_rows):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.show()

