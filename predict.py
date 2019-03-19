import torch
import json

from get_args import get_args
from data_processing import process_image
from model_management import setup_model, validation

def main():
    '''predict the type of flower in an image'''
    #get prediction arguments
    args = get_args('predict')
    
    #define model and load checkpoint
    model, criterion, optimizer, epochs = setup_model(args.arch, args.checkpoint)
    
    #process input image for network
    image = process_image(args.input)

    #process image
    image = process_image(args.input)
    image = torch.from_numpy(image)
    
    #add dimension to fit model's expecation as we are only sending one image
    image = image.unsqueeze_(0).type(torch.cuda.FloatTensor)

    #eval mode
    model.eval()
    
    if args.gpu == 'gpu':
        device = 'cuda'
        print("Using GPU")
    else:
        device = 'cpu'
        print("WARNING, USING CPU")
    
    #pass tensor to model
    image, model = image.to(device), model.to(device)
    output = model.forward(image)

    #take exponential to get probabilities
    output = torch.exp(output)

    #extract classes and probs
    indexes = torch.topk(output, args.top_k)[1][0]
    probs = torch.topk(output, args.top_k)[0][0]

    #move indexes to probs to cpu and convery to array
    indexes, probs = indexes.to('cpu'), probs.to('cpu')
    indexes, probs = indexes.detach().numpy(), probs.detach().numpy()
    
    #invert dictionary to do lookups
    inv_class_to_idx = {i: j for j, i in model.class_to_idx.items()}

    #convert indexes to classes
    classes = []
    for ii in indexes:
        classes.append(str(inv_class_to_idx[ii]))

    #import class dictionary 'cat_to_name.json'
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    #convert classes to names
    names = []
    for ii in classes:
        names.append(cat_to_name[ii]);
    
    for ii in range(args.top_k):
        print(names[ii], probs[ii])

if __name__ == "__main__":
    main()