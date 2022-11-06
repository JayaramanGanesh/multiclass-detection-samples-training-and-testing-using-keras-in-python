from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os

#loadng the file and loadeding the weights
jsonfile = open("tarinined json file ","r")
loaded_model_json = jsonfile.read()
jsonfile.close()
model = model_from_json(loaded_model_json)
model.load_weights("tarinined json file .h5")
print("model loaded successfully")

#classify the image samples
def classify(img_file):
    imgname = ima_file
    testimg = image.load_img(image,target_size = (128,128))
    testimg = image.img_to_array(testimg)
    testimg = np.expand_dims(testimg, axis = 0)
    result = model.predict(testimg)
    classes = ["this field also how many classes for the model in weights"]
    label2 = classes[result.argmax()]
    print(label2,imgname)


#get the testing samples 
path = "testing samples path"
files =[]
for r, d, f in os.walk(path):
    for file in f :
        if ".jpg" in file:
            print(files)
            files.append(os.path.json(r,file))

#printing the  testing the samples
for f in files:
    classify(f)
    print("\n")