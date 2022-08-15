from data_loading import *
from keras import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras import layers
from keras.layers import Sequential

output_classes=26 #A-Z

#create model function
def create_model(num_layers=4):

  model = Sequential()
  num_filters = [64, 128, 128, 256, 512]

  for layer in range(num_layers):
    model.add(Conv2D(filters = num_filters[layer], kernel_size = 5, padding = 'same', activation = 'relu'))
    if layer != num_layers:
      model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

  model.add(Flatten())
  model.add(Dense(output_classes, activation='softmax'))

  return model
