import tensorflow as tf
import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
from tensorflow import keras
from keras.backend import clear_session
from keras.layers import Dense
from keras.models import Model
from keras.applications import *
from time import perf_counter

from base_data_loading import *
from base_train import *
from base_test import *
from base_main import *

def get_model(model):

    ## Load the pretained base model
    kwargs =    {'input_shape':(128, 128, 3),
                'include_top':False,
                 'weights': 'imagenet',
                'pooling':'max'}
    
    model = model(**kwargs)
    model.trainable = False
    
    inputs = model.input

    x = Dense(128, activation='elu')(model.output)
    x = Dense(128, activation='elu')(x)

    outputs = Dense(26, activation='softmax')(x)

    new_model = Model(inputs=inputs, outputs=outputs)

    new_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return new_model

def pretrain_train(pt_train,pt_val,models):
        
        ##train the models
    for name, model in models.items():
    
        ## Get the model
        m = get_model(model['model'])
        models[name]['model'] = m
    
        start = perf_counter()

        ##Fit the model
        history = m.fit(pt_train,validation_data=pt_val,epochs=5,verbose=0)

        ##Save the duration and the val_accuracy
        duration = perf_counter() - start
        duration = round(duration,2)
        models[name]['perf'] = duration
        print(f"{name:20} trained in {duration} sec")

        val_acc = history.history['val_accuracy']
        models[name]['val_acc'] = [round(v,4) for v in val_acc]
    return models

def compare_models():

    data_path = split_data("comparison")
    train_aug,val_aug= set_data_augmentation()
    pt_train,pt_val,pt_test = load_image_generator(train_aug,val_aug,105,(128,128),data_path)
    ##get keras models
    models = {
    "InceptionResNetV2": {"model":tf.keras.applications.InceptionResNetV2, "perf":0},
    "InceptionV3": {"model":tf.keras.applications.InceptionV3, "perf":0},
    "ResNet101": {"model":tf.keras.applications.ResNet101, "perf":0},
    "ResNet101V2": {"model":tf.keras.applications.ResNet101V2, "perf":0},
    "ResNet152": {"model":tf.keras.applications.ResNet152, "perf":0},
    "ResNet152V2": {"model":tf.keras.applications.ResNet152V2, "perf":0},
    "ResNet50": {"model":tf.keras.applications.ResNet50, "perf":0},
    "ResNet50V2": {"model":tf.keras.applications.ResNet50V2, "perf":0},
    "VGG16": {"model":tf.keras.applications.VGG16, "perf":0},
    "VGG19": {"model":tf.keras.applications.VGG19, "perf":0},
    "Xception": {"model":tf.keras.applications.Xception, "perf":0}
    }

    print("Training the models with weights of the pretrained Inception, Resnet, VGG and Xception Models.")
    models = pretrain_train(pt_train,pt_val,models)

    ##Create a DataFrame with the results
    models_result = []

    for name, v in models.items():
        models_result.append([ name, models[name]['val_acc'][-1], 
                                models[name]['perf']])

    df_results = pd.DataFrame(models_result, 
                                columns = ['model','val_accuracy','Training time (sec)'])
    df_results.sort_values(by='val_accuracy', ascending=False, inplace=True)
    df_results.reset_index(inplace=True,drop=True)
    df_results.to_csv("/content/results/pretrained_models/pretrained_models_results.csv")
    
    ##save plot for the accuracy after 5 epochs
    fig_path = "/content/results/pretrained_models/pretrained_model_acc_5_epochs.jpg"
    plt.figure(figsize = (15,5))
    sns.barplot(x = 'model', y = 'val_accuracy', data = df_results, color="red")
    plt.title('Accuracy on the Test set after 5 epochs', fontsize = 15)
    plt.ylim(0,1)
    plt.xticks(rotation=90)
    plt.savefig(fig_path)

    ## Train data on the best architecture - in this case - ResNet5OV2
    best_pretrained_model = df_results.iloc[0]

    print("Results of training after 5 epochs on all models saved in results. Best model is "+ str(best_pretrained_model[0]))

    ## Fine tuning the pretrained model
    pretrained_model = get_model( eval("tf.keras.applications."+ best_pretrained_model[0]) )

    #initialize augmentation and image generators and load all data with 75-15-10 split
    train_aug,val_aug= set_data_augmentation()
    train,val,test = load_image_generator(train_aug,val_aug,105,(128,128),(project_path+"/data/base_data"))

    ##train the best pretrained model
    history = pretrained_model.fit(train,
                        validation_data=val,
                        epochs=15,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                patience=3,
                                restore_best_weights=True)]
                        )
    train_results(history,"comparison")

    ## Run model on test data and save results
    test_accuracy = test_model(pretrained_model,test,"comparison")
    print("-------Results of comparison with pretrained models saved in ../results/pretrained_models/")