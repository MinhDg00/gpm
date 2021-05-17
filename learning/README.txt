SET UP:
----------------------------------------------------------------------------------------------------------------------
1.Install numpy 

$pip install numpy
$pip3 install numpy

2.Copy hw5-data folder into HW4 folder

FILE DESCRIPTION
----------------------------------------------------------------------------------------------------------------------

1. main.py: running script
2. parameter_learning: contains all 3 learning algorithms FOD-learn, POD-EM-learn, Mixture-Random-Bayes
3. bayesian_model.py: contain bayesian network and conditional probability table object
4. helper.py: contain misc script (eg. Read data, find mean, find log difference)


INSTRUCTIONS: 

----------------------------------------------------------------------------------------------------------------------

1. Unzip HW4 Folder

2. Launch Command Prompt/Terminal and navigate to the HW4 folder.

3. Input the following command:

   $python3 main.py uai_file task_id train_file test_file 

4. The program will ask user to input the w, N and type of proposal distribution

EXAMPLE: python3 main.py 1.uai 1 train-f-1.txt test.txt

NOTE: 
1. When input 1/2/3.uai, program will automatically consider only consider files inside dataset 1/2/3. For example, 'train-f-1.txt' will be in the path 'hw5-data/dataset1/train-f-1.txt'

2. For task 3, user also need to input k 

python3 main.py 1.uai 3 train-f-1.txt test.txt
--------------------------------------------------
Input number of mixture Bayesian networks: 2




