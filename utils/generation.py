import os
import torch
import gdown
import logging
import langid
langid.set_languages(['en', 'zh', 'ja'])

import pathlib
import platform
if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system().lower() == 'linux':
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer

from macros import *

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

url = 'https://drive.google.com/file/d/10gdQWvP-K_e1undkvv0p2b7SU6I4Egyl/view?usp=sharing'

checkpoints_dir = "./checkpoints/"

model_checkpoint_name = "checkpoint.pt"

model = None

codec = None

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

def preload_models():
    global model, codec
    if not os.path.exists(checkpoints_dir): os.mkdir(checkpoints_dir)
    if not os.path.exists(os.path.join(checkpoints_dir, model_checkpoint_name)):
        gdown.download(id="10gdQWvP-K_e1undkvv0p2b7SU6I4Egyl", output=os.path.join(checkpoints_dir, model_checkpoint_name), quiet=False)
    # VALL-E
    model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    ).to(device)
    checkpoint = torch.load(os.path.join(checkpoints_dir, model_checkpoint_name), map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.eval()

    # Encodec
    codec = AudioTokenizer(device)

@torch.no_grad()
def generate_audio(text, prompt=None, language='auto', accent='no-accent'):
    global model, codec, text_tokenizer, text_collater
    text = text.replace("\n", "").strip(" ")
    # detect language
    if language == "auto":
        language = langid.classify(text)[0]
    lang_token = lang2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # load prompt
    if prompt is not None:
        if not os.path.exists(prompt):
            prompt = "./presets/" + prompt + ".npz"
        if not os.path.exists(prompt):
            raise ValueError(f"Cannot find prompt {prompt}")
        prompt_data = np.load(prompt)
        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]

        # numpy to tensor
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    else:
        audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32).to(device)
        text_prompts = torch.zeros([1, 0]).type(torch.int32)
        lang_pr = lang if lang != 'mix' else 'en'

    enroll_x_lens = text_prompts.shape[-1]
    logging.info(f"synthesize text: {text}")
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens
    # accent control
    lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=-100,
        temperature=1,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang,
    )
    samples = codec.decode(
        [(encoded_frames.transpose(2, 1), None)]
    )

    return samples[0][0].cpu().numpy()