from base_architecture import *

from tensorflow import keras
from keras import callbacks
import matplotlib.pyplot as plt


##run the model with the hyperparameters of the ongoing trial
def train_CNN(model,trial_no,train,val):
  ##implement callback function
  early_stopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", 
                                        patience=5, restore_best_weights = True) 
  #train the model, get results, and save it
  history = model.fit(x=train, validation_data = val, epochs=30, shuffle = True, verbose = 1, callbacks=[early_stopping])
  save_train_results(history,trial_no)
  model.save("../results/hyperparameter_optimization/trial_models/model_"+trial_no+".h5")
  return model
    
#training results
def train_results(history,trial_no):

  fig_path = "../results/hyperparameter_optimization/trial_plots/"+"trial"+str(trial_no)+".jpg"
  plt.subplot(2,1,1)
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0, 1])
  plt.legend(loc='lower right')
  plt.title('Accuracy Curve')

  plt.subplot(2,1,2)
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label = 'val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.ylim([0, 10])
  plt.legend(loc='upper right')
  plt.title('Loss Curve')
  
  plt.rcParams["figure.figsize"] = (10,10)
  plt.tight_layout()
  plt.rcParams['savefig.orientation'] = 'landscape'
  plt.savefig(fig_path)
  ##plt.show() currently commented out because this is for .ipynb and not python file