import torch
import data_handler as dh
import numpy as np


# load the model
model = torch.load('transformer_model1.pth')

# get the vocab from the data_handler
_, _, _, vocab = dh.get_data()

# assign the vocab index of the word
blue = vocab['blue']
yellow = vocab['yellow']
car = vocab['car']


# define the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# putting the index to a tensor
blue = torch.tensor(blue).to(device)
yellow = torch.tensor(yellow).to(device)
car = torch.tensor(car).to(device)

# get the embeddings of the word the word
blue = model.encoder(blue)
yellow = model.encoder(yellow)
car = model.encoder(car)


# put the tensor into numpy array (they also need to be in the cpu)
blue = blue.detach().cpu().numpy()
yellow = yellow.detach().cpu().numpy()
car = car.detach().cpu().numpy()

# np.inner returns the cosine distance, that we use to measure the semantic similarity
print("Blue and yellow: ", np.inner(blue, yellow))
print("Blue and car: ", np.inner(blue, car))
print("Blue and blue: ", np.inner(blue, blue))
