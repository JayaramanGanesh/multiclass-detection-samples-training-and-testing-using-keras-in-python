from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing import ImageDataGenerator

# neural network preparation
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation = "relu", input_shap = (128,128,3)))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size= (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size= (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size= (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size= (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(6, activation = "softmax"))
model.compile(optimizer="adam",loss = "categorical_crossentropy",metrics=["accurcy"])

#training the neural network
train_datagen = ImageDataGenerator(rescale = None,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
testdatagen = ImageDataGenerator(rescale = 1./255)
trainingset = train_datagen.flow_from_directory("train",target_size = (128,128), batch_size = 32, class_mode = "categorical")
labels = (trainingset.class_indices)
testset = testdatagen.flow_from_directory("test",target_size = (128,128), batch_size = 32, class_mode = "categorical")
labels2 =(testset.class_indices)
model.fit_generator(trainingset,steps_per_epoch = 375, epochs = 5, validation_data = testset,validation_steps= 125)

#finally trained model save the json file
model_json = model.to_json()
with open ("model2.json","w") as json_file:
    json_file.write(model_json)
    model.save_weights("model2.h5")
    print("saved model to disk")
