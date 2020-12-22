# PU-HIV

## Folders
- data contains the biological network data used in the paper.
- python contains the python scripts for Biased-SVM.
- results contains the experimental results of the python script running.
- sample contains some training and testing examples.
- images contains some extended pictures in the paper.


## Usage
1. prepare the training and testing datasets by following the format in Sample folder
2. run python \main.py to run Biased-SVM. run the command "python main.py -c 6 [-train 301] [-test 301].
   The first parameter is the feature type of Biased-SVM.  
   The second paramter is the train set(optional parameters), sample/train is selected by default, you can also choose the five independent datasets(the 301, 746, 1625, impens and schilling datasets respectively).  
   The last one is is the test set(optional parameters), sample/test is selected by default, you can also choose the five independent datasets(the 301, 746, 1625, impens and schilling datasets respectively).
3. Possible values for the feature types of Biased-SVM are listed as below.
   - orthogonal coding = 0
   - chemical coding = 1
   - Evocleave coding = 2
   - orthogonal coding + chemical coding = 3
   - chemical coding + Evocleave coding = 4
   - orthogonal coding + Evocleave coding = 5
   - orthogonal coding + chemical coding + Evocleave coding = 6
4. use the data in the sample for training and testing, you must first run the command "java -jar lib/EvoCleave.jar Sample 16 0" to generate the corresponding Evocleave co-evolution patterns.

## Node: The codes should be compatible with Python 3.6 and Java 1.8. If you get errors when running the scrips, please try the recommended versions.
