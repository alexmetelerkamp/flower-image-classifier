import argparse

def get_args(mode):
    parser = argparse.ArgumentParser()
    
    if mode == 'train':
        parser.add_argument('data_directory', type = str, default = '/flowers', 
            help = 'Path to the folder of training images') 
        
        parser.add_argument('--checkpoint', type = str, default = 'Checkpoints', 
            help = 'Path to the folder of checkpoints') 

        parser.add_argument('--arch', type = str, default = 'densenet121', 
            help = 'Architecture of choice, densenet121 or vgg13') 
        
        parser.add_argument('--learning_rate', type = float, default = 0.001, 
            help = 'Learning rate') 
        
        parser.add_argument('--hidden_units', type = int, default = 512, 
            help = 'Classifier hidden units') 
        
        parser.add_argument('--epochs', type = int, default = 1, 
            help = 'Training epochs')
        
        parser.add_argument('--gpu', type = str, default = 'gpu', 
            help = 'Use GPU')
    elif mode == 'predict':
        parser.add_argument('input', type = str, default = 'mags.jpg', 
            help = 'Path to image to predict')

        parser.add_argument('checkpoint', type = str, default = 'Checkpoints', 
            help = 'Path to the folder of checkpoints') 
        
        parser.add_argument('--top_k', type = int, default = 3, 
            help = 'Return top K most likely classes')

        parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
            help = 'JSON mapping of categories to flower names')
        
        parser.add_argument('--gpu', type = str, default = 'gpu', 
            help = 'Defaults to GPU usage, use "cpu" if you do not want GPU utilised')
        
        parser.add_argument('--arch', type = str, default = 'densenet121', 
            help = 'Architecture of choice, densenet121 or vgg13') 
    else:
        print('No inputs defined')
    
    arguments = parser.parse_args()    
    print(arguments)
    
    return arguments

if __name__ == "__main__":
    arguments = []
    get_args('train')
   