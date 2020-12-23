# Predicting HIV-1 protease cleavage sites with positive-unlabeled learning

#### Zhenfeng Li
#### Lun Hu
#### Zehai Tang
#### Cheng Zhao
------
## Folders
- `Datasets` contains all the five datasets used in the experiments.
- `Python` contains the python scripts of PU-HIV.
- `EvoCleave` contains the executable file of EvoCleave.
- `Sample` contains the sample data fo training and testing .


## Usage
1. prepare the training and testing datasets by following the format in Sample folder
2. run `java -jar EvoCleave/EvoCleave.jar Sample 16 0` to exract coevolutionary patterns. Note that PU-HIV will automatically use the coevolutonary patterns extracted from positive set for feature vector construction.
2. run `python main.py` to execute PU-HIV. Several parameters have to be predetermined.  
   `-f`: the features used t construct feature vectors, possible values of this parameter are 0 (AAI), 1 (CheP), 2 (CoP), 3 (AAI+CheP), 4 (CheP+CoP), 5 (AAI+CoP) and 6 (AAI+CheP+CoP);  
   `-c1`: the value of C1;  
   `-beta`: the value of beta;  
   `-i`: input folder;  
   `-cv`: optional, PU-HIV will switch to cross validation mode if provided and the value of this parameter is the number of folds in cross validation.  
   Hence, a complete command to run the sampl data is `python main.py -f 6 -c1 8 -beta 2 -i ../Sample`.  
   If the paratemter `cv` is provided, the subfolders in the input folder should be named with integers. For example, if the value of `cv` is set as 10, the names of subfolders should be from 1 to 10.
3. check out the results.txt file in the input folder for the prediction results of testing data.


Node: The codes should be compatible with Python 3.6 and Java 1.8. If you get errors when running the scrips, please try the recommended versions.
