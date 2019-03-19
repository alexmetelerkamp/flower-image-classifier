import torch
from torch import nn, optim
from torchvision import models

from collections import OrderedDict

def setup_model(arch='densenet121', checkpoint='/Checkpoints', learning_rate=0.001, hidden_units=512, train_set=0):
    model_loading = True
    
    '''define neural network and load last checkpoint'''
    if arch == 'densenet121':
        model = models.densenet121(pretrained = True)     
        classifier_input = 1024
    elif arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        classifier_input = 25088
    else:
        print('Architecture not found')
        
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    #redefine classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_input, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units,102)),
                          ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier.dropout = nn.Dropout(p=0.3)
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

    path = str(checkpoint+'/'+arch+'.pth')

    if model_loading:
        #load model checkpoint if it exists
        print('Attempting to load: ', path)
        try:
            state = torch.load(path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state['state_dict'])
            model.class_to_idx = state['class_to_idx']
            optimizer.load_state_dict(state['optimizer'])
            epochs = state['epochs']
            print('Successfully loaded: ', path)
        except Exception as e: 
            print(e)
            print('No matching checkpoint file found, so loading untrained network')
            model.class_to_idx = train_set.class_to_idx
            epochs = 1
    else:
        print('Model checkpoint loading is off, so loading untrained network')
        model.class_to_idx = train_set.class_to_idx
        epochs = 1
        
    
    return model, criterion, optimizer, epochs

def validation(model, valid_loader, criterion, device):
    '''test current parameters on validation set'''
    valid_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            
            images, labels, model = images.to(device), labels.to(device), model.to(device)
            
            output = model.forward(images)

            valid_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            
            equality = (labels.data == ps.max(dim=1)[1])
            
            accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy
    
def save_model(epochs, model, optimizer, checkpoint):
    '''save model params and state'''
    
    state = {'epochs': epochs,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'class_to_idx' : model.class_to_idx}
    
    try:
        torch.save(state, checkpoint)
    except:
        print('Failed to save model state: ', checkpoint)