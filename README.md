# Vignesh-Kashinath-lab

## Installation
#### Jupyter Notebook
<ul>


<li>
  Create Python Environment
</li>

<li>
  Install Jupyter Notebook (https://jupyter.org/install)
</li>

<li>
  Clone the Repository
</li>

<li>
   Open Terminal
</li>

<li>
  Activate the environment
</li>

<li>
  install requirements file
</li>


<li>
  Launch Jupyter Notebook
</li>

<li>
  Open the CompleFlow.ipynb file  
</li>

<li>
  Run each cell either using "shift  + enter" or using the play icon in the task bar
</li>
</ul>

## Things to Remember
<ul>

  <li>
    If you are running the Python file, install the requirements file using pip install -r requirements.txt
  </li>

  <li>
      Parameters to be changed:
      <ul>
        <li>
          basedir (change this to the path where all things are stored)
        </li>
        <li>
          train_imgs, train_masks with appropriate folders
        </li>
        <li>
          Annotations on Napari or some tool
        </li>
        <li>
          10-20 images, with background as 0, 1 - structure1, 2, structure2, ...
        </li>
        <li>
          masks.tif (all the 10-20 images)
        </li>
      </ul>
  </li>

  <li>
    Keep image size atleast 256 x 256
  </li>
</ul>
## Code Explanation
<ul>
  <li>
    Library and Module Imports
  </li>
  <li>
    Setting Base Directory for all the resources
  </li>

  <li>
    Separate variables for train images and masks
  </li>

  <li>
    since we are using a segmentation model, we are splitting the images into patches of size 256 x256 with a step size of 64 x 64
  </li>

  <li>
    preprocessing to clean the data so that the model works better, the steps include edge detection and histogram equalization
  </li>

  <li>
    Train/Test split
  </li>

  <li>
    Defining Data Loaders so that large datasets are loaded into the memory with ease
  </li>

  <li>
    Definining the model
  </li>

  <li>
    Setting Hyperparameters (Try changing learning rate )
  </li>

  <li>
    Setting the training device (works faster on GPU)
  </li>

  <li>
    Training 
  </li>

  <li>
    Inference using postprocessing by combining the same labels on each slice (image) stacked together to generate a 3D structure
  </li>
</ul>

