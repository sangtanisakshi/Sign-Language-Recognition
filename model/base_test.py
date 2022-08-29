import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report,accuracy_score

from base_train import *
from base_architecture import *
from base_main import project_path

##run test on the model with the current trial or the final model
def test_model(current_model,test,trial_no="NA"):
    pred = current_model.predict(test, batch_size=(test.samples//test.batch_size+1))
    pred_vals = np.argmax(pred, axis=1)
    score = accuracy_score(test.classes,pred_vals)
    if trial_no == "best_model":
        test_results(test,pred_vals)
    if trial_no == "comparison":
        test_results(test,pred_vals,True)
    return score

##get test results and save them
def test_results(test,pred_vals,comparison=False):

    if comparison==True:
        fig_path=(project_path + "/results/pretrained_models/best_pt_model_confusion_matrix.jpg")
        cr_path=(project_path + "/results/pretrained_models/best_pt_model_evaluation_metrics.txt")
    else:
        fig_path=(project_path + "/results/best_model/figures/confusion_matrix.jpg")
        cr_path=(project_path + "/results/best_model/figures/evaluation_metrics.txt")
    
    classes = list(test.class_indices.keys())
    cm = confusion_matrix(test.classes,pred_vals)
    cm_txt = np.array2string(cm)
    disp = ConfusionMatrixDisplay(cm, display_labels = classes)
    plt.rcParams["figure.figsize"] = (15,10)
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.savefig(fig_path)
    #plt.show() commented for ipynb later
    #print(classification_report(test.classes, pred_vals, target_names=classes))
    
    cr = classification_report(test.classes, pred_vals, target_names=classes)
    ##save metrics in text file
    f = open(cr_path, 'w')
    f.write('Metrics for Best Model\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))
    f.close()

   