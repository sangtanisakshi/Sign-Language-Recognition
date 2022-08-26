import tensorflow
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.models import Sequential,Model
from keras.losses import CategoricalCrossentropy
from keras.optimizer_v2 import adam

NUM_CLASSES=26 #A-Z

def create_model(num_layers,activation,learning_rate):
    ##creating the model architecture
    model = Sequential()
    num_filters = [64, 128, 128, 128, 256, 256, 512, 512]

    for layer in range(num_layers):
        model.add(Conv2D(input_shape= (64,64,3), filters = num_filters[layer], kernel_size = 5, padding = 'same', activation = activation))
        if layer != num_layers:
            model.add(MaxPooling2D())
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    #compile the model
    model.compile(loss=CategoricalCrossentropy(),
                  optimizer=adam.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    print("--------"+"New CNN Model Created"+"--------")
    return model