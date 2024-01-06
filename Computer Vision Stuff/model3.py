# code copied and adapted from keras image classification tutorial at
# https://www.tensorflow.org/tutorials/images/classification

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt

import pathlib

data_dir = pathlib.Path('./Photos')

batch_size = 16

'''
note: I think this is rather large for image size, but I was looking at some of
the smaller images after resizing and it seemed to average the colors together
which made the band colors pretty much disappear. Perhaps zooming in more on the
original picture might help
'''

img_height = 512
img_width = img_height // 2 # I think this is closer to the aspect ratio

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=2, # seed for shuffle
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=2, # for shuffle
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

# I'm not really sure if the caching the caching is better or not...
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

############################################################################
# IMAGE AUGMENTATION:

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip(mode="horizontal_and_vertical",
                      input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(factor=0.2, # maximum of 20% rotation on image
                          fill_mode = 'nearest'),
    layers.RandomZoom(height_factor=(-.1, 0), # 0 - 10% zoom in
                      width_factor = None,
                      fill_mode = 'nearest'),
  ]
)

display_num = 0 # displays this number of augmented images for you to see
for _ in range(display_num):
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()

############################################################################
# MODEL:
    
num_classes = len(class_names)

model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255), # rescales pixels from [0, 255] to [0, 1]
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 32
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# uncomment to plot training data

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

############################################################################
# PREDICTIONS:

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