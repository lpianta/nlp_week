from transformers import pipeline
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np
from punctuator import Punctuator
import jamspell

# load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
punctuator = Punctuator("../models/INTERSPEECH-T-BRNN2.pcl")
corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel("../models/spellchecker_en.bin")

# load any audio file of your choice
speech, rate = librosa.load("../10mintest.mp3", sr=16000)
lenght = librosa.get_duration(speech, sr=16000)
n_chuncks = np.ceil(lenght / 10)
chuncks = np.array_split(speech, n_chuncks)


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


text = transcriptor(chuncks)
# print(text)
text = text.lower()
text = punctuator.punctuate(text)
text = corrector.FixFragment(text)
print(text)
