from base_architecture import *

from tensorflow import keras
from keras import callbacks
import os
import matplotlib.pyplot as plt


#train the CNN model 
def train_CNN(model,train,val,EPOCHS,trial_no="NA"):

  ##implement callback function
  early_stopping = callbacks.EarlyStopping(monitor="val_loss", mode="min", 
                                        patience=5, restore_best_weights = True)

  ##train the model, get results, and save it
  history = model.fit(x=train, validation_data = val, epochs=EPOCHS, shuffle = True, verbose = 1, callbacks=[early_stopping])

  if trial_no == "best_model":
    model.save("results/best_model/"+trial_no)
    train_results(history,trial_no)
  
  return model,history
    
#save training plots as results
def train_results(history,trial_no):

  if trial_no == "best_model":
    fig_path = ("results/best_model/loss_acc_curve_" + trial_no + ".jpg")
  else:
    fig_path = ("results/hyperparameter_optimization/trials_plots/" + "trial" + str(trial_no) + ".jpg")
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