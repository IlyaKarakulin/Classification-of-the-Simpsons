# Classification-of-the-Simpsons

My first computer vision project is the Classification of the Simpsons. The idea of the task was taken from the competition https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset. 

## Dataset

The dataset offered in the competition is severely unbalanced. Some classes have 3 pictures. 

![screenshot](./img/balans.jpg)

There have been attempts to train a model on such data, but it is almost impossible to test such a model, since it will not be possible to divide it into a validation and training sample for each class.



gdown --folder https://drive.google.com/drive/folders/10ET4wN898yG2oiRshYxTH95p-Dpoor1S?usp=sharing

tar tar -xJf ./simpsons_dataset/dataset.tar.xz


tensorboard --logdir=meta_data