# flower-image-classifier

## Summary
A flower image classifier, using DenseNet121 and VGG13

- train.py can be used to train a new model (either VGG13 or DenseNet121) to identify images of flowers. 
- predicy.py is used to predict the classfication of a new flower image. 

Both VGG and DenseNet models are pretrained on ImageNet, and further flower-specific training is done on the VGG 102 'Category Flower Dataset'  http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html 

## Usage

### Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
1. Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
2. Choose architecture: python train.py data_dir --arch "vgg13" or "densenet121"
3. Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
4. Use GPU for training: python train.py data_dir --gpu "gpu" or "cpu" (GPU by default)

### Predict flower name from an image with predict.py along with the probability of that name. 

Basic usage: python predict.py /path/to/image checkpoint
Pass in a single image /path/to/image to return the flower name and class probability.

Options:
1. Return top KK most likely classes: python predict.py input checkpoint --top_k 3
2. Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
3. Use GPU for inference: python predict.py input checkpoint --gpu
