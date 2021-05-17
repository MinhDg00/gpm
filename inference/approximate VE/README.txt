INSTRUCTIONS: 

I have 2 script part6_12.py and part6_3.py. 

- part6_12.py tests first 2 part of question 6. You can input any w, N and options to use different proposal distribution

- part6_3.py run the script to get table result for question 3. Each time script will give me the result of 1 PGM 

----------------------------------------------------------------------------------------------------------------------

A. Running part6_12.py: 

1. Launch Command Prompt/Terminal and navigate to the HW2 folder.

2. Input the following command:

   $python3 part6_12.py inputFilePath evidFilePath

3. The program will ask user to input the w, N and type of proposal distribution

Example: python3 test.py Grids_14.uai Grids_14.uai.evid
Please enter w-cutset bound integer: 1
Please enter number of samples you want to use: 200
Input proposal distribution option you want to use (adaptive/uniform): adaptive
The probability of evidence or partion function using adaptive proposal distribution is: 371.54752152704924
It take 3.7944042682647705s to run

----------------------------------------------------------------------------------------------------------------------

B. Running part6_3.py

1. Launch Command Prompt/Terminal and navigate to the HW2 folder. NOTE: .uai should be in the same folder. If not, you might need to change the file inside the script

2. Input the following command:

   $python3 part6_3.py

3. The program will output a dictionary of error and time wrt to that inputFilePath. 

----------------------------------------------------------------------------------------------------------------------
