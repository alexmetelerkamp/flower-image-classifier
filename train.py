import torch

from get_args import get_args
from data_processing import setup_training
from model_management import setup_model, validation, save_model

def main():
    '''train neural network to identify flower types'''
    print_every = 50
    steps = 0

    #get training args
    args = get_args('train')
    
    #setup training data
    train_loader, valid_loader, test_loader, train_set = setup_training(args.data_directory)
    
    #define model and load checkpoint
    model, criterion, optimizer, epochs = setup_model(args.arch, args.checkpoint, args.learning_rate, args.hidden_units, train_set)
    
    if args.gpu == 'gpu':
        device = 'cuda'
        print("Using GPU")
    else:
        device = 'cpu'
        print("WARNING, USING CPU")
        
    epochs = args.epochs
    
    model.to(device)
        
    #run training epochs
    for e in range(epochs):
        #zero out the loss for an epoch
        running_loss = 0

        #run training on train_loader
        for idx, (images, labels) in enumerate(train_loader):
            #count batches
            steps += 1

            images, labels = images.to(device), labels.to(device)
                       
            #clear gradients for each pass
            optimizer.zero_grad()

            outputs = model.forward(images)

            #calculate loss
            loss = criterion(outputs, labels)

            #accumulate gradients
            loss.backward()

            #update parameters with current gradients
            optimizer.step()

            #accumulate running loss
            running_loss += loss.item()

            #update screen
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                path = str(args.checkpoint+'/'+args.arch+'.pth')
    
                save_model(epochs, model, optimizer, path)
    
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion, device)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
   

#if __name__ == "__main__":
main()