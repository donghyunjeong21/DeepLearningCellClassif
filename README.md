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

Place all your images to be trained into a single directory. These should all have the same file type (i.e. .tif), and only images that are to be used in training should be placed in this directory. 

Initialize the program. Click "Train your own model" button. A new page will open. Select the directory that contains the training images and type the file type, exluding the period (i.e. tif instead of .tif).

Select the relevant parameters: 

Choose the channel to be used as input to the model. This will be the channel the model looks at to predict cell characteristics. (NOTE: all channels are 0-indexed. this means first channel is referred to as 0)

Choose the ground truth channel. This is most likely a fluorescent image channel whose intensity corresponds to the characteristic of the cell you're trying to predict. 

Choose the segmentation channel. This is an optional fluorescent channel that can be used to segment the cells. This can be useful if your cells are particularly difficult to segment or the image quality in the input channel is bad. Type -1 to indicate no segmentation channel, in which case the input channel will be used for segmentation. 

Use GPU: click to use GPU. Must be set up to use relevant packages with GPU. This is strongly suggested.

Cell diameter: Estimated diameter of your cell in number of pixels. 

Threshold: for binary classification, select the average intensity level in the fluorescent channel that will classify the cell as positive. For continuous prediction, input -1. 

Model name: name of the model to be saved. 

Press the "Train" button to initiate training. The program will notify you of the number of cells it identified across all the training images. If the threshold value was supplied, it will indicate how many of these cells are positively labeled. If this is satisfactory, press "Continue."

The model will train itself and save as "providedmodelname.h5." Along with this file, there will be an additional metadata.txt file that must be kept in the same directory as the h5 file. 
