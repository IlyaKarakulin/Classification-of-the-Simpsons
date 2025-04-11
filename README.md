# Classification-of-the-Simpsons

My first computer vision project is the Classification of the Simpsons. The idea of the task was taken from the competition https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset. 

## Dataset

The dataset offered in the competition is severely unbalanced. Some classes have 3 pictures. 

![screenshot](./img/balans.jpg)

There have been attempts to train a model on such data, but it is virtually impossible to test such a model because it would not be possible to separate it into a validation and training sample for each class. \ 
Balancing the dataset with augmentation is not a good idea because training will be the same as validation. It will not allow to evaluate the model's ability to classify simpsons in other scenes, and will not help to detect overfitting. \
Therefore, new images were found for each class. A minimum of 80 images for training and 20 for validation. Then the set was augmented to 1163 for training and 290 for validation for each class. As a result, we obtained a balanced dataset in which the images with validation are different from the training images. \

it's been published on google drive and can be downloaded using command.

```bash 
pip install gdown
gdown --folder https://drive.google.com/drive/folders/10ET4wN898yG2oiRshYxTH95p-Dpoor1S?usp=sharing
```

tar tar -xJf ./simpsons_dataset/dataset.tar.xz


tensorboard --logdir=meta_data