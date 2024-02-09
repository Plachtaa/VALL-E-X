import os
import torch
import torchaudio
import logging
import langid
import whisper

langid.set_languages(['en', 'zh', 'ja','ar'])

import numpy as np
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from utils.g2p import PhonemeBpeTokenizer

from macros import *

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
if torch.backends.mps.is_available():
    device = torch.device("mps")
codec = AudioTokenizer(device)

if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
whisper_model = None

@torch.no_grad()
def transcribe_one(model, audio_path):
    # load audio and pad/trim it to fit 30 seconds
    try:
        audio = whisper.load_audio(audio_path)
        # Continue with audio processing
    except FileNotFoundError:
        logging.error(f"File not found at path: {audio_path}")
    except Exception as e:
        logging.error(f"An error occurred during audio processing: {e}")
   
   
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

    text_pr = result.text
    if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
        text_pr += "."
    return lang, text_pr




def make_transcript(name, wav, sr, transcript=None):
    # Validate input parameters
    assert isinstance(wav, torch.Tensor), "Input 'wav' must be a torch.FloatTensor"
    assert isinstance(sr, int), "Sample rate 'sr' must be an integer"
    
    if wav.abs().max() > 1:
        wav /= wav.abs().max()
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    assert wav.ndim and wav.size(0) == 1, "Input 'wav' must have correct dimensions"

    if transcript is None or transcript == "":
        logging.info("Transcript not given, using Whisper...")
        global whisper_model
        if whisper_model is None:
            whisper_model = whisper.load_model("medium", download_root=os.path.join(os.getcwd(), "whisper"))
        whisper_model.to(device)
        try:
            lang, text = transcribe_one(whisper_model, "D:\\MachineCourse\\dataset\\train")
            if lang is not None:
                lang_token = lang2token.get(lang, "")  # Use get() method to handle missing key
                text = lang_token + text + lang_token
            else:
                text = ""  # Assign an empty string if lang is None
        except Exception as e:
            logging.error(f"Error during audio transcription: {e}")
            text = ""
        finally:
            whisper_model.cpu()
    else:
        text = transcript
        lang, _ = langid.classify(text)
        lang_token = lang2token.get(lang, "")  # Use get() method to handle missing key
        text = lang_token + text + lang_token

    torch.cuda.empty_cache()
    return text, lang
