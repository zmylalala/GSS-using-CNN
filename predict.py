import tensorflow as tf
from PIL import Image
import numpy as np


trash_dict = {0: 'It is a cardboard', 1: 'It is a glass', 2: 'It is a metal',
              4: 'It is a paper', 5: 'It is a plastic', 6: 'It is a regular trash'}

path = './pics/glass/glass0.jpg'
img = Image.open(path)
img = img.resize((180, 180))
img = np.asarray(img)
img = (np.expand_dims(img, 0))
model = tf.keras.models.load_model('model/saved_model')
prediction = model.predict(img)
print(trash_dict[np.argmax(prediction[0])])

