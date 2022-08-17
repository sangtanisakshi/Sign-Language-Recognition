from base_architecture import *
from base_train import *
from base_data_loading import *
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from keras.backend import clear_session

from base_test import test_model

def hpo(trial):

    IMG_TARGET_SIZE,EPOCHS = set_fixed_hyperparameters()
    train_aug,val_aug= set_data_augmentation()
    
    clear_session()
    trial_no = trial.number

    #get the hyperparameters for the ongoing trial
    batch_size,num_layers,activation,learning_rate = get_hpo_parameters(trial)

    #get the image directories with the trial's selected batch size
    train,val,test = load_image_generator(train_aug,val_aug,batch_size,IMG_TARGET_SIZE)

    #create model architecture with the current trial's hyperparameters
    model = create_model(num_layers,activation,learning_rate)

    #train the model and get the model
    current_model,history = train_CNN(model,train,val,EPOCHS)

    #save loss and accuracy plots for the current trial training
    train_results(history,trial_no)

    #run the model on the test data and get test accuracy
    score = test_model(current_model,test)
    #return accuracy of this trial to the study
    return score