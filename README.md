# Sign-Language-Recognition

# 1. Getting Started
  Using CNNs to statically and dynamically recognize visual sign language gestures and convert it to text and audio. 

# 1.1 Prerequisities

- Python 3.7 or higher
- CUDA-enabled GPU (optional, but highly recommended)


# 1.2 Project Code Folder Structure:
Project folder consists of 3 files, which are:
  - model
  - results
  - requirements.txt

# 1.3 Project Explanation
Project model is placed in its respective directory under "model" directory.
Model folder consists of 7 files, which are:

1.	base_main.py – This is the main program where model run start. 
-	First of all, Pass the base data and tfl data as parameter on run_model function. 
-	After that, unzip the base data and split them. 
-	Assign image size and epochs by set_fixed_hyperparameters functions and train and validation augmentation data from set_data_augmentation function. 
-	Then run the hyperparameters optimization using hyperparameter optimization framework optuna, where create the study form optuna and pass hpo which has the hyperparameter optimization function. 
-	We have 25 trials which also pass as parameter on study. 
-	After finishing every trial, Choose the best parameter and train the model with it. 
-	After getting the base model, compare it with keras pre-trained models.
2.	base_architecture.py – create the CNN model architecture and compile and return it
3.	data_loading.py – To read the data and preprocess them. The respective data read and preprocess tasks are unzip data, split data, set fixed hyperparameters, generate training data with data augmentation, loading data, and defining hyperparameters that need on tuning.
4.	compare_models.py – From this file, we load the pretrained model and train it. After that, compare it with 11 best keras model. Comparison keras models are – 
-	IncetionResNetV2
-	InceptionV3	
-	ResNet101
-	ResNet101V2
-	ResNet152
-	ResNet152V2
-	ResNet50
-	ResNet50V2
-	VGG16
-	VGG19
-	Xception

Create a dataframe from model result. Find the best pre-trained model. Train the data with best pre-trained model architecture and create a new model with it. After that, run the model and save the results. 

5.	base_hyperparameter_optimization.py – Using this function, get the accuracy score for every optuna study trials and return accuracy of this trials to the study. 	
6.	base_train.py – Here, train the CNN model and save training plots on results folder.  
7.	base_train.py – In this file, Run test on the model. Get the results and save them on results folder.

# 1.4 Running the codes
1. Create virtualenv and activate it.

2. Install dependencies from requirements.txt file.
  
   ``` pip install -r requirements.txt   ```
   
3. Download dataset using the command and save it in the root folder:
   ``` gdown 1iJ7lx0x6pEM9IqbScxAYvtfWtGnXt6A9 ```
   
4. Trained Model file can be downloaded here (To be saved in ./results/best_model/):
``` gdown 1DWKm0Qv0f3H6tPN_J84vh0xI9y_sP_Nk ```
   
4. To run the model:
  ``` python ./model/base_main.py  ```
