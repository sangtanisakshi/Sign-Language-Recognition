import splitfolders
import tensorflow as tf
import warnings
import optuna
import os
from tensorflow import keras
from keras import preprocessing
from keras.preprocessing.image import ImageDataGenerator

from base_main import project_path

def get_data():
    #unzip the downloaded dataset
    data=(project_path+"/asl-alphabet.zip")
    from zipfile import ZipFile
    with ZipFile(data,"r") as zipObj:
        zipObj.extractall(path=(project_path+"/data/"))

def split_data(data_type="base_data"):

    os.rename((project_path+"/data/asl_alphabet_train/"),(project_path+"/data/dataset/"))
    #remove the unwanted classes and folders
    os.rmdir(project_path+"/data/asl_alphabet_test/")
    os.rmdir(project_path+"/data/dataset/space/")
    os.rmdir(project_path+"/data/dataset/delete/")
    os.rmdir(project_path+"/data/dataset/nothing/")


    ip_path = (project_path+"/data/dataset/")

    if data_type == "base_data":
        op_path = (project_path+"/data/base_data/")
        SPLIT_RATIO=(0.75,0.15,0.10)

    elif data_type == "comparison":
        op_path = (project_path+"/data/pt_data/")
        SPLIT_RATIO = (0.10,0.05,0.85)
    
    splitfolders.ratio(input=ip_path, output=op_path,
    seed=42, ratio=SPLIT_RATIO, group_prefix=None, move=False)
    print("Split data in the ratio, "+ str(SPLIT_RATIO) +" for training, validation and test.")

    #now that the data is split, we dont need to keep the entire original dataset
    return op_path
    
def set_fixed_hyperparameters():
    #fixed hyperparameters
    IMG_TARGET_SIZE = (64,64)
    EPOCHS = 30
    print("Fixed hyperparameters:")
    print("Image target size: ", IMG_TARGET_SIZE)
    print("Maximum training epochs: ", EPOCHS)
    return IMG_TARGET_SIZE,EPOCHS

def set_data_augmentation():
    #image data generators for training set with data augmentation
    gen_aug = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=15) 
    #image data generator for validation and test set without 
    gen = ImageDataGenerator(rescale=1./255)

    print("Training data augmentation set with rescaling, horizontal flip and rotation.")
    print("Validation and test data augmentation set with rescaling.")

    return gen_aug,gen

def load_image_generator(train_aug,val_aug,batch_size,IMG_TARGET_SIZE,data_path=(project_path+"/base_data/")):
    ##image directories - loading data into train, validation and test
    train = train_aug.flow_from_directory(data_path+"train", 
                                target_size=IMG_TARGET_SIZE,
                                batch_size=batch_size, shuffle=True)

    val = val_aug.flow_from_directory(data_path+"val",
                              target_size=IMG_TARGET_SIZE,
                              batch_size=batch_size, shuffle=False)

    test = val_aug.flow_from_directory(data_path+"test", 
                               target_size=IMG_TARGET_SIZE, 
                               class_mode=None, shuffle=False)
    
    return train,val,test

def get_hpo_parameters(trial):
    
    #defining the hyperparameters that need tuning
    batch_size = trial.suggest_int('batch_size',64,128)
    num_layers = trial.suggest_categorical('num_layers', [3,4,5,6])
    activation = trial.suggest_categorical('activation',['relu','selu','elu'])
    learning_rate = trial.suggest_float("learning_rate",1e-5,1e-2,log=True)
    print("Hyperparameters for current trial selected.")
    return batch_size,num_layers,activation,learning_rate