import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("mood_model.keras")

classes = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

img_path = "test.jpg"

img = image.load_img(
    img_path,
    target_size=(48, 48),
    color_mode="grayscale"
)

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

result = classes[np.argmax(prediction)]

print("Detected Expression:", result)