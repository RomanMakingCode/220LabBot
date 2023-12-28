import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_testing_labels(main_directory):
    testing_labels = []
    #Iterate through each subdirectory in the main directory
    for subdir in os.listdir(main_directory):
        subdir_images = []
        subdir_path = os.path.join(main_directory, subdir)
        
        #Check if the path is indeed a directory
        if os.path.isdir(subdir_path):
            #Add all .jpg files in the subdirectory to the list
            subdir_images.extend([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.jpg')])

        #Add corresponding labels to respective lists
        for i in range(len(subdir_images)):
            testing_labels.append(str(subdir))

    return testing_labels

photos_directory = os.path.join('.', 'Photos')

# Create Image Data Generators for train and test sets
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25) # using validation_split to split data

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    photos_directory,
    target_size=(256, 256), # smaller size for practical purposes
    batch_size=32,
    class_mode='categorical',
    subset='training' # set as training data
)

validation_generator = train_datagen.flow_from_directory(
    photos_directory, # same directory as training data
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation' # set as validation data
)

# Model architecture
model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(256, 256, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dropout(.5),

    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(9, activation = 'softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=5, validation_data=validation_generator)

# Evaluate the model

test_photos_directory = os.path.join('.', 'TestPhotos')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_photos_directory,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict
predictions = model.predict(test_generator)

Testphotosdirectory = os.path.join('.', 'TestPhotos')
correct_labels = get_testing_labels(Testphotosdirectory)

labels = ["Brown Black Bronw Gold", "Brown Black Orange Gold", "Brown Black Red Gold", "Brown Black Yellow Gold", "Red Red Orange Gold", 
          "Red Red Red Gold", "Yellow Purple Black Gold", "Yellow Purple Brown Gold", "Yellow Purple Red Gold"]

count = 0
for p in predictions:
    max_index = np.argmax(p)
    print("Model : " + labels[max_index] + ", Correct : " + correct_labels[count])
    count = count + 1

print('done')