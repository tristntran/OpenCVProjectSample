This project is an example script for loading in testing data from the COCO database and using it to test the DarkNet Implementation of
YOLOv4.

# Project Structure
the `load_samples.py` script is used to build a directory full of COCO formatted images. You are also able to add in your files.
If they are not coco formatted, then you will not be able to calculate accuracy.

By running the main script `main.py`, you are able to classify images in the "Images directory." 

# Requirements
This project depends on the [fiftyone](https://voxel51.com/docs/fiftyone/getting_started/install.html)
and [opencv](https://opencv-tutorial.readthedocs.io/en/latest/) 
libraries. Install these before running the project.