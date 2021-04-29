from transformers import pipeline
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np


# load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# load any audio file of your choice
speech, rate = librosa.load("jocko.m4a", sr=16000)
print(type(speech))
print(len(speech))

# speech = speech[:len(speech)//12]  # cutting to only ten seconds
chuncks = np.array_split(speech, 12)
# print(chuncks.shape)


def transcriptor(chunks):
    string = ""
    for i in chuncks:
        input_values = tokenizer(i, return_tensors='pt').input_values
        # Store logits (non-normalized predictions)
        logits = model(input_values).logits
        # Store predicted id's
        predicted_ids = torch.argmax(logits, dim=-1)
        # decode the audio to generate text
        transcriptions = tokenizer.decode(predicted_ids[0])
        string += transcriptions + " "
    return string


#text = transcriptor(chuncks)
# print(text)
