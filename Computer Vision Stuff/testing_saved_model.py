import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

import pathlib

# This script gives more information about how the model is making predictions
# Tests against every image in 'Photos'

data_dir = pathlib.Path('./Photos')

batch_size = 16
img_height = 512
img_width = img_height // 2

# only used to get class names
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=2,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

model = tf.keras.models.load_model('saved_model.keras')

print('\nTesting the model:\n')

photos_directory = pathlib.Path('./Photos')

tf.keras.utils.disable_interactive_logging()
# tf.keras.utils.enable_interactive_logging()

predictions = dict()
for category in class_names:
    predictions[category] = 0

correct_predictions = dict()
for category in class_names:
    correct_predictions[category] = 0

folder_list = photos_directory.iterdir()

total = 0
for folder in folder_list:
    if not folder.is_dir():
        continue

    photo_list = folder.iterdir()
    resistor_type = folder.name

    subtotal = 0
    subtotal_correct = 0
    avg_confidence = 0.0
    for photo in photo_list:

        img = tf.keras.utils.load_img(
            photo, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        results = model.predict(img_array)
        score = tf.nn.softmax(results[0])

        prediction = class_names[np.argmax(score)]
        predictions[prediction] += 1

        confidence = 100 * np.max(score)
        avg_confidence += confidence

        if prediction == resistor_type:
            subtotal_correct += 1
            correct_predictions[prediction] += 1
        
        subtotal += 1

        # uncomment to list each prediction with confidence value
        print(f'Prediction: {prediction} ({confidence:.2f}% confidence)')

    print(f'Category: {resistor_type}:')
    print(f'subtotal score = {subtotal_correct}/{subtotal} = {subtotal_correct / subtotal * 100:.2f}%')
    print(f'average confidence = {avg_confidence / subtotal:.2f}%\n')
    total += subtotal

print('===========================================================\nSummary:\n')
total_correct = 0
for category in class_names:
    total_correct += correct_predictions[category]
    print(f'Predictions for {category} = {predictions[category]} ({predictions[category]/total*100:.2f}%)')
    if predictions[category] == 0:
        print(f'Zero predictions for {category}\n')
        continue
    print(f' - {correct_predictions[category]/predictions[category]*100:.2f}% of these were correct\n')

print(f'total score = {total_correct}/{total} = {total_correct / total * 100:.2f}%\n')

print('done')