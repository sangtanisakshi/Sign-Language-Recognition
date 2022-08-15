##TODO
#change grassknot zip name to base dataset or sth

import splitfolders
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator


def get_data(data):
#unzip the downloaded dataset
    from zipfile import ZipFile
    with ZipFile("grassknot.zip","r") as zipObj:
        zipObj.extractall()


##initialize values
split_ratio = (0.75,0.15,0.10)
img_target_size = (64,64)
data_batch_size = 128

def split_data(ratio=split_ratio):
    splitfolders.ratio("./grassknot/", output="./data/",
    seed=42, ratio=split_ratio, group_prefix=None, move=False)

    #image data generators for training set with data augmentation
    gen_aug = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=15) 

#image data generator for val and test set
    gen = ImageDataGenerator(rescale=1./255) 

    train = gen_aug.flow_from_directory("./data/train",
                                target_size=img_target_size,
                                batch_size=data_batch_size, shuffle=True)

    val = gen.flow_from_directory("./data/val",
                              target_size=img_target_size,
                              batch_size=data_batch_size, shuffle=False)

    test = gen.flow_from_directory("./data/test", 
                               target_size=img_target_size, 
                               class_mode=None, shuffle=False)

##print some images from the train set