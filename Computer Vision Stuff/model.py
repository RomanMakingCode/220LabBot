import os
from PIL import Image
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

'''
This script pulls all the images from the subfolders
3/4 are used to train the model
1/4 are used to test the model's accuracy
'''

'''
This function goes into the subdirectories from the given directory and returns 4 lists
First two is the training_set images (their directory) and that image's corresponding label
Second two is the testing_set images (their directory) and that image's corresponding label
'''
def split_images(main_directory):
    testing_set = []
    testing_labels = []
    training_set = []
    training_labels = []
    all_labels = []

    #Iterate through each subdirectory in the main directory
    for subdir in os.listdir(main_directory):
        subdir_images = []
        subdir_path = os.path.join(main_directory, subdir)
        
        #Check if the path is indeed a directory
        if os.path.isdir(subdir_path):
            all_labels.append(subdir_path)
            #Add all .jpg files in the subdirectory to the list
            subdir_images.extend([os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.jpg')])
        
        #Split into training and testing sets
        subdir_split = int(len(subdir_images) * 0.75)
        training_set.extend(subdir_images[:subdir_split])
        testing_set.extend(subdir_images[subdir_split:])

        #Add corresponding labels to respective lists
        for i in range(subdir_split):
            training_labels.append(str(subdir))
        for i in range(len(subdir_images) - subdir_split):
            testing_labels.append(str(subdir))

    return  training_set, training_labels, testing_set, testing_labels, all_labels


#Open directory and get photos
photos_directory = os.path.join('.', 'Photos')
training_images, training_labels, testing_images, testing_labels, all_labels = split_images(photos_directory)

'''
#ensure matching labels to photos
print("training set")
for i in range(len(training_set)):
    print(training_set[i] + " | " + training_labels[i])

print("testing set")
for i in range(len(testing_set)):
    print(testing_set[i] + " | " + testing_labels[i])
'''

#This function will normalize the rgb values
datagen = ImageDataGenerator(rescale=1./255) 

#Format Training data
training_set =[]
for i in training_images:
    image = load_img(i, target_size = (256,256))
    image_array = img_to_array(image)
    image_array = image_array.reshape((1,) + image_array.shape)
    image_array = datagen.flow(image_array, batch_size=1).next()
    training_set.append(image_array)

#Format Testing data
testing_set = []
for i in testing_images:
    image = load_img(i, target_size = (256,256))
    image_array = img_to_array(image)
    image_array = image_array.reshape((1,) + image_array.shape)
    image_array = datagen.flow(image_array, batch_size=1).next()
    testing_set.append(image_array)

print(testing_set[0].shape)
#Shove it into a model
model = keras.Sequential([

    keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (256,256,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),
    keras.layers.Dropout(.5),

    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(len(all_labels), activation = 'softmax')

])

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(training_set, training_labels, epochs=5)

output = model.predict(testing_images)

print(output == testing_labels)


print('done')
