# Predicting HIV-1 protease cleavage sites with positive-unlabeled learning

#### Zhenfeng Li
#### Lun Hu
#### Zehai Tang
#### Cheng Zhao
------
## Folders
- `Dataset` contains all the five datasets used in the experiments.
- `Python` contains the python scripts of PU-HIV.
- `EvoCleave` contains the executable file of EvoCleave.
- `Sample` contains the sample data fo training and testing .


## Usage
1. prepare the training and testing datasets by following the format in Sample folder
2. run `java -jar EvoCleave/EvoCleave.jar Sample 16 0` to exract coevolutionary patterns, note that only coevolutonary patterns extracted from positive set should be used.
2. run `python main.py -c 6 [-train 301] [-test 301]` to execute PU-HIV
   The first parameter is the feature type of Biased-SVM.  
   The second paramter is the train set(optional parameters), sample/train is selected by default, you can also choose the five independent datasets(the 301, 746, 1625, impens and schilling datasets respectively).  
   The last one is is the test set(optional parameters), sample/test is selected by default, you can also choose the five independent datasets(the 301, 746, 1625, impens and schilling datasets respectively).
3. possible values for the feature types of Biased-SVM are listed as below.
   - orthogonal coding = 0
   - chemical coding = 1
   - Evocleave coding = 2
   - orthogonal coding + chemical coding = 3
   - chemical coding + Evocleave coding = 4
   - orthogonal coding + Evocleave coding = 5
   - orthogonal coding + chemical coding + Evocleave coding = 6

Node: The codes should be compatible with Python 3.6 and Java 1.8. If you get errors when running the scrips, please try the recommended versions.
