# Model_Cascade_Optimization

## Exploring the Computational Cost of Machine Learning at the Edge for Human-Centric Internet of Things

Repository containing the material for the analysis of the paper "XX".

### Dataset
* Office  Hydration  Monitoring  (OHM)  Dataset

This dataset focuses on classifying office employees' hydration patterns based on an accelerometer and a gyroscope sensor placed on different liquid containers (e.g., mug, bottles or glasses). It contains 1000 instances performed by 10 subjects and includes 25 variations of different interactions that could be made with with liquid containers. Those interactions are grouped into three main classes: (1) drinking from a bottle, (2) drinking from a cup and, (3) other kinds of interactions). 

### Structure of this repository

* `~/Classification_results.ipynb`    : Jupyter Notebook to obtain the classification results for the reference model, stacking techniques as well as the sequential, parallel and hybrid implementations of the model cascade.
* `~/OHM_Dataset`    : Original dataset 
* `~/Timing`: Python scripts for obtaining the results of the experiments

  * `~/OHM_Dataset_train_test`    : Dataset with a pre-set train-test split to perform the timing comparison
    * `TRAIN` :  800 Instances
    * `TEST` :  200 Instances
  * `Time_Reference_Model.py` :  Save data processing times for inference when clasifying all the instances of the Test folder with the reference model. It will generate a .txt file containing average and std values 
  * `Time_Model_Cascade.py ` :  Save data processing times for inference when clasifying all the instances of the Test folder with the parallel cascade method. It will generate a .txt file containing average and std values 


### Reproducing the experiments

1. Install the dependencies `pip install -r requirements.txt`
2. Execute the Jupyter Notebook to obtain the classification results.
3. Run the selected script included on `~/Timing` to obtain perform the timing experiments (see *Notes* below). 


### Notes

All the experiments were done for the referenced dataset and the scripts are specifically created to process it. To use a different dataset it is necessary to adapt the scripts and dataset. 

`Time_` staring scrips must be run using `ipython3`. Example: `ipython3 Time_Model_Cascade.py` 


In the very first lines of some of the script we have included several variables:

* `root` determines the dataset folder, `OHM_Dataset` for the regular dataset and `OHM_Dataset_Train_Test` for the splited version
* `number_initial_components` selects the number of signal components to consider (9 Max). Please, note that changing the number of features affects the maximun number of features. 
* `best_features_number` allows selecting the number of features to consider from the 486 initial subset
* `featMn` The number of features considered in every level of the cascade
* `Threshold_X` The log loss value that sets the confidence threshold to accept or not a prediction at each stage of the cascade
* `number_of_segments` The number of times every window of data is divided
* `n_splits` The number of folds for the n-splits-folf Cross Varidation Process
* `n_times` Represents the number of times the cross-validation process is performed to obtain the average f1 results (Classification results)

