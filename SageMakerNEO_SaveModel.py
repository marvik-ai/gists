from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input
model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
model.save('InceptionV3.h5')
