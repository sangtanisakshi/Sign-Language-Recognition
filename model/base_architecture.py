from keras import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras import layers
from keras.layers import Sequential
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

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
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    
    return model

