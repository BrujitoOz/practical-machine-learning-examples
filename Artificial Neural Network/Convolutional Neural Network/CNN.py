#%% Library and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#%% Init CNN
classifier = Sequential()
#%% Conv.
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))
#%% Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
#%% Flattening
classifier.add(Flatten())
#%% Full Connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid"))
#%% Compile CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
#%% Fit CNN to the images to train 
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit_generator(training_dataset,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=testing_dataset,
                        validation_steps=2000)

# %%
