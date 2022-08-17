from base_data_loading import *
from base_test import *
from base_hyperparameter_optimization import *
from compare_models import *
from transfer_learning import *

import optuna
import joblib
from optuna.samplers import TPESampler
from types import SimpleNamespace

def run_model(data="../base_data.zip"):
    
    get_data(data)
    split_data()
    IMG_TARGET_SIZE,EPOCHS = set_fixed_hyperparameters()
    train_aug,val_aug= set_data_augmentation()
    
    #run hyperparameter optimization
    sampler = TPESampler(seed=123)  # Make the sampler behave in a deterministic way and get reproducable results
    study = optuna.create_study(direction="maximize",sampler=sampler)
    study.optimize(hpo(train_aug,val_aug,IMG_TARGET_SIZE), n_trials=2)
    joblib.dump(study,'../results/hyperparameter_optimization/trial_study_data/study.pkl')

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
    #train model with best hyperparameters and save results
    best_model,history = train_CNN(best_model,trial_no,train,val)

    #test model on the test data, get the accuracy and save metrics
    test_accuracy = test_model(best_model,test,"best_model")
    print("Best model ")
    
    

if __name__ == "__main__":
    run_model()
