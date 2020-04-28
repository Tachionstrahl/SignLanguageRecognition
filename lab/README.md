# Files and subdirectories in this folder

...and their meaning.

Future work:
 - [ ] Clean up and refactor notebooks

## Jupyter Notebooks

[Jupyter](https://jupyter.org/install) is required for working with jupyter notebooks.

### [Train.ipynb](Train.ipynb)

This is a notebook for loading the csv files in the subfolder [data](data) and training a model.
The output should be `.h5` files within this folder.

### [Data Analysis.ipynb](Data%20Analysis.ipynb)

This is a notebook for analysing and visualizing hand and face detection data from the csv files in the subfolder [data](data).

## Subfolders

### [data](data)

Contains csv files in subdirectories. Each subdirectory has the name of a sign (ref. german sign language).
Every row in a csv file translates to a frame of its equivalent video file, where minimum a face and one hand is detected by MediaPipe.

### [assets](assets)

Contains graphics that are included in notebooks.
