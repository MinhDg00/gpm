SET UP:
----------------------------------------------------------------------------------------------------------------------
1.Install numpy 
```shell
$pip install numpy
$pip3 install numpy
``` 

FILE DESCRIPTION
----------------------------------------------------------------------------------------------------------------------

1. main.py: running script
2. parameter_learning: contains all 3 learning algorithms FOD-learn, POD-EM-learn, Mixture-Random-Bayes
3. bayesian_model.py: contain bayesian network and conditional probability table object
4. helper.py: contain misc script (eg. Read data, find mean, find log difference)


INSTRUCTIONS: 

----------------------------------------------------------------------------------------------------------------------

1. Unzip Folder

2. Launch Command Prompt/Terminal and navigate to the  folder.

3. Input the following command:
```shell
   $python3 main.py uai_file task_id train_file test_file 
```
4. The program will ask user to input the w, N and type of proposal distribution

EXAMPLE: 
```shell
python3 main.py 1.uai 1 train-f-1.txt test.txt
```
