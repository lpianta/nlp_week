from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from pathlib import Path
import requests
from transformers import pipeline
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from punctuator import Punctuator
import jamspell
import numpy as np
import os


UPLOAD_FOLDER = "./temp"

# if not os.path.exists(UPLOAD_DIRECTORY):
#    os.makedirs(UPLOAD_DIRECTORY)

template_dir = Path("../templates")
static_dir = Path("../static")
app = Flask(__name__, template_folder=str(template_dir),
            static_folder=str(static_dir))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
punctuator = Punctuator("../models/INTERSPEECH-T-BRNN2.pcl")
corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel("../models/spellchecker_en.bin")


def transcriptor():
    string = ""
    chuncks = get_chuncks()
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


def get_chuncks():
    speech = get_audio()
    lenght = librosa.get_duration(speech, sr=16000)
    n_chuncks = np.ceil(lenght / 10)
    chuncks = np.array_split(speech, n_chuncks)
    return chuncks


def get_audio():
    audio = request.files["file_1"]
    filename = secure_filename(audio.filename)
    audio.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    speech, rate = librosa.load("./temp/" + filename, sr=16000)
    os.remove("./temp/" + filename)
    return speech


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/file_upload", methods=["POST"])
def post_audio():
    text = transcriptor()
    text = text.lower()
    text = punctuator.punctuate(text)
    text = corrector.FixFragment(text)
    return render_template("transcription.html", response=text)


if __name__ == "__main__":
    app.run(debug=True)
