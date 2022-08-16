from base_architecture import *
from base_train import *
from base_data_loading import *

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from keras.backend import clear_session

def hpo(trial,train_aug,val_aug,IMG_TARGET_SIZE):

    trial_no = trial.number
    clear_session()
    
    #get the hyperparameters for the ongoing trial
    batch_size,num_layers,activation,learning_rate = get_hpo_parameters(trial)

    #get the image data generators with the trial's image target size and batch size
    train,val,test = load_image_generator(train_aug,val_aug,IMG_TARGET_SIZE,batch_size)

    #create model architecture with the current trial's hyperparameters
    model = create_model(trial_no,num_layers,activation,learning_rate)

    #train the model and return the history
    train_trial = train_CNN(model,trial_no,train,val)

    #run the model on the test data and get test accuracy
    pred = train_trial.predict(test, batch_size=(test.samples//test.batch_size+1))
    pred_vals = np.argmax(pred, axis=1)
    score = accuracy_score(test.classes,pred_vals)

    #return accuracy of this trial to the study
    return score