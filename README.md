# DeepLearningCellClassif

<b>Authors:</b> Donghyun Paul Jeong and Maksym Zarodiuk 

<b>Affiliation:</b> University of Notre Dame, Bioengineering Graduate Program

<b>Introduction:</b>


This is a machine learning tool that allows users to easily train their own AI algorithm to recognize cell morphological features. The AI, once trained, can be used to predict expression levels of certain proteins or classify the cell types based on non-fluorescent images of cell morphologies. The program contains three parts: first, the images are preprocessed in a manner of user's choice. Then the images are segmented to identify the location of each cell, then the segmented images of each cell are run through a convolutional neural network, using a user-supplied fluorescent channel as the target values. 

<b>Installation: </b>


Download all the script files in this directory onto a local directory. The files must remain in the same directory. On command line, run main_GUI.py, and the user interface will open. Alternatively, the code can be run from command line by calling on main_driver.py. 

The program requires the following packages that may need to be installed in addition to standard Python 3.10.10 libraries:

NumPy >= 1.22.3

Matplotlib >= 3.7.0

Cellpose >= 2.0.5

Scikit-image >= 0.19.3

Scikit-learn >= 1.2.1

Tensorflow >= 2.11.0

Keras >= 2.11.0

PyTorch (CUDA version if running with GPU) >= 1.13.1

Imageio >= 2.25.1


<b>Training your own model:</b>

