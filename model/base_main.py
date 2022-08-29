global project_path

import optuna
import joblib
import os
from optuna.samplers import TPESampler
from types import SimpleNamespace

from base_data_loading import *
from base_test import *
from base_hyperparameter_optimization import *
from compare_models import *

def run_model():
    
    ##unzip the data 
    get_data()

    ##split the data into training, validation and test and get the path where the data is stored
    ##data type can be base_data, comparison or tfl_data
    data_path = split_data("base_data")

    ##set fixed hyperparameteres and data augmentation 
    IMG_TARGET_SIZE,EPOCHS = set_fixed_hyperparameters()
    train_aug,val_aug= set_data_augmentation()

    ##run hyperparameter optimization
    sampler = TPESampler(seed=123)  # Make the sampler behave in a deterministic way and get reproducable results
    study = optuna.create_study(direction="maximize",sampler=sampler)
    study.optimize(hpo, n_trials=25)
    joblib.dump(study,(project_path+"/results/hyperparameter_optimization/trials_data/study.pkl"))
    study_data = pd.DataFrame(study.trials_dataframe())
    data_csv = study_data.to_csv((project_path + "/results/hyperparameter_optimization/trials_data/study.csv"))

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial params:")
    for key, value in study.best_params.items():
        print(" {}: {}".format(key, value))
    
    #get best trial hyperparameters and train the model with that
    best_params = SimpleNamespace(**study.best_params)

    ##get the image directories with the best trial's batch size
    train,val,test = load_image_generator(train_aug,val_aug,best_params.batch_size,IMG_TARGET_SIZE)

    ##create model architecture with the best trial's hyperparameters
    best_model = create_model(best_params.num_layers,best_params.activation,best_params.learning_rate)

    trial_no = "best_model"
    ##train model with best hyperparameters and save results
    print("Got the best model hyperparameters. Training the model now.")
    best_model,history = train_CNN(best_model,train,val,EPOCHS,trial_no)

    ##test model on the test data, get the accuracy and save metrics
    test_accuracy = test_model(best_model,test,"best_model")
    print("Best model tested on test data and saved")
    
    ##train pretrained models with our base dataset
    print("---------Training our base model data on pretrained Keras Imagenet Models---------")

    compare_models()

if __name__ == "__main__":

    #get os path
    project_path = os.getcwd()
    run_model()
