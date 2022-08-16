from base_train import *
from base_architecture import *
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report


##run test on the model with the best trial

##get test results and save them
def test_results(test,y_vals):
    classes = list(test.class_indices.keys())
    cm = confusion_matrix(test.classes,y_vals)
    disp = ConfusionMatrixDisplay(cm, display_labels = classes)
    plt.rcParams["figure.figsize"] = (15,10)
    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.savefig('../results/hyperparameter_optimization/trial_results/'+'confusion_matrix.jpg')
    plt.show()
    print(classification_report(test.classes, y_vals, target_names=classes))