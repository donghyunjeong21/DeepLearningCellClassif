# DeepLearningCellClassif

Authors: Donghyun Paul Jeong and Maksym Zarodiuk 
Affiliation: University of Notre Dame, Bioengineering Graduate Program

Introduction:

This is a machine learning tool that allows users to easily train their own AI algorithm to recognize cell morphological features. The AI, once trained, can be used to predict expression levels of certain proteins or classify the cell types based on non-fluorescent images of cell morphologies. The program contains three parts: first, the images are preprocessed in a manner of user's choice. Then the images are segmented to identify the location of each cell, then the segmented images of each cell are run through a convolutional neural network, using a user-supplied fluorescent channel as the target values. 

Installation: 

Download all the script files in this directory onto a local directory. The files must remain in the same directory. On command line, run main_GUI.py, and the user interface will open. Alternatively, the code can be run from command line by calling on main_driver.py. 

The program requires the following packages that may need to be installed in addition to standard Python3 libraries:
NumPy
Matplotlib 
Cellpose
Scikit-image
Tensorflow
Keras
PyTorch (CUDA version if running with GPU)

Training your own model:

