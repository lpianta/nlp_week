# importing libraries
import io
import torch
import torchtext
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter


# parameters
bptt = 10
batch_size = 10
eval_batch_size = 10

tokenizer = get_tokenizer('basic_english')


def preprocess(iterator, vocab):
    data = [torch.tensor([vocab[word] for word in tokenizer(
        line)], dtype=torch.long) for line in iterator]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data, batch_size):
    n_batch = data.size(0) // batch_size
    # putting all the data together
    data = data.narrow(0, 0, n_batch*batch_size)
    # actually creating the batches reshaping data, contigous is to mantain the order
    data = data.view(batch_size, -1).t().contiguous()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device)


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def get_data():
    train_iter = WikiText2(split='train')  # download the train iterator
    counter = Counter()                    # instantiate a Counter istance
    # update the counter with the tokens (kind of a dictionary)
    for line in train_iter:
        counter.update(tokenizer(line))
    vocab = Vocab(counter)                 # create a Vocab from the counter

    train_iter, val_iter, test_iter = WikiText2()

    train_data = preprocess(train_iter, vocab)
    val_data = preprocess(val_iter, vocab)
    test_data = preprocess(test_iter, vocab)

    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    return train_data, val_data, test_data, vocab


# Take a look at the tokens in vocab
#train_data, val_data, test_data, vocab = get_data()
#print("Boats: ", vocab["boats"])
#print("Swan: ", vocab["swan"])
# take a look at the shape of a batch
#batch = get_batch(train_data, 0)
#print("Batch shape: ", batch[0].shape)
# take a look at the batchified data
#batchify = batchify(train_data, 10)
#print("Batchified data: ", batchify.size())
