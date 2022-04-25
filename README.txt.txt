Steps to run the code :

1. Please ensure that general libraries like numpy, pandas, matplotlib and installed and configured.
2. Additionally pygame library and ipython will be required to run the code.
    -- to install pygame use the following commands based on your python environment
          pip install pygame          pip install ipython
                OR                            OR
          conda install pygame        conda install ipython
4. Make sure to open the folder ML_Project containing the python code in your editor. When unzipping the folder, some times an ML_Project folder might be created inside another ML_Project folder.
   opening the wrong ML_Project folder will result in misconfiguration of paths used to load resources for teh game.
5. to run the game, run the agent.py file.

Steps to change the parameters :
1. Changing learning rate : learning rate can be changed by modifying the value of the LEARNING_RATE variable on line 16

2. Changing hidden layers: We have 2 Nueral Network implentations. Which one to pick can be configured by commenting out either line 25 or 26.
    a) NeuralNetwork (line 25) can be configured with 1 hidden layer. change the number of HIDDEN_NEURONS on line 19 to update the value.
    b) NeuralNetwork2 (line 25) can use 1 or 2 hidden layers. To configure with one hidden layer, give HIDDEN_NEURONS an integer value, else assign a list with 2 values.

3. Changing the activation function: we have 3 activation functions implemented. They can be assigned by changing ACTIVATION on line 20 with the following values ('tanh', 'relu', 'sigmoid')
