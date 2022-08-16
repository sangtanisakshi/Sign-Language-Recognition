from base_data_loading import *
from base_test import *
from base_hyperparameter_optimization import *
from compare_models import *
from transfer_learning import *

import optuna
import joblib
from optuna.samplers import TPESampler

def run_model(data="C:/Users/cupca/Desktop/Uni Study Material/Semester VI/Deep Learning/ASLproject/base_data.zip"):
    
    get_data(data)
    split_data()
    IMG_TARGET_SIZE,EPOCHS = set_fixed_hyperparameters()
    train_aug,val_aug= set_data_augmentation()
    
    #run hyperparameter optimization
    sampler = TPESampler(seed=123)  # Make the sampler behave in a deterministic way and get reproducable results
    study = optuna.create_study(direction="maximize",sampler=sampler)
    study.optimize(hpo(train_aug,val_aug,IMG_TARGET_SIZE), n_trials=25)
    joblib.dump(study,'../results/hyperparameter_optimization/trial_results/study.pkl')

    #get best trial and 

    

if __name__ == "__main__":
    run_model()
