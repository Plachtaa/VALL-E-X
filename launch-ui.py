# coding: utf-8
import argparse
import logging
import os
import pathlib
import time
import tempfile
import platform
import webbrowser
import sys
print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
print(f"You are using Python version {platform.python_version()}")
if(sys.version_info[0]<3 or sys.version_info[1]<7):
    print("The Python version is too low and may cause problems")

if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import langid
langid.set_languages(['en', 'zh', 'ja'])

import nltk
nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

import torch
import torchaudio
import random

import numpy as np

from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer
from descriptions import *
from macros import *
from examples import *

import gradio as gr
import whisper
from vocos import Vocos
import multiprocessing

thread_count = multiprocessing.cpu_count()

print("Use",thread_count,"cpu cores for computing")

torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

# VALL-E-X model
if not os.path.exists("./checkpoints/"): os.mkdir("./checkpoints/")
if not os.path.exists(os.path.join("./checkpoints/", "vallex-checkpoint.pt")):
    import wget
    try:
        logging.info("Downloading model from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt ...")
        # download from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt to ./checkpoints/vallex-checkpoint.pt
        wget.download("https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
                      out="./checkpoints/vallex-checkpoint.pt", bar=wget.bar_adaptive)
    except Exception as e:
        logging.info(e)
        raise Exception(
            "\n Model weights download failed, please go to 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'"
            "\n manually download model weights and put it to {} .".format(os.getcwd() + "\checkpoints"))

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
    )
checkpoint = torch.load("./checkpoints/vallex-checkpoint.pt", map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys
model.eval()

# Encodec model
audio_tokenizer = AudioTokenizer(device)

# Vocos decoder
vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

# ASR
if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
try:
    whisper_model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
except Exception as e:
    logging.info(e)
    raise Exception(
        "\n Whisper download failed or damaged, please go to "
        "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
        "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

# Voice Presets
preset_list = os.walk("./presets/").__next__()[2]
preset_list = [preset[:-4] for preset in preset_list if preset.endswith(".npz")]

def clear_prompts():
    try:
        path = tempfile.gettempdir()
        for eachfile in os.listdir(path):
            filename = os.path.join(path, eachfile)
            if os.path.isfile(filename) and filename.endswith(".npz"):
                lastmodifytime = os.stat(filename).st_mtime
                endfiletime = time.time() - 60
                if endfiletime > lastmodifytime:
                    os.remove(filename)
    except:
        return

def transcribe_one(model, audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
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

def make_npz_prompt(name, uploaded_audio, recorded_audio, transcript_content):
    global model, text_collater, text_tokenizer, audio_tokenizer
    clear_prompts()
    audio_prompt = uploaded_audio if uploaded_audio is not None else recorded_audio
    sr, wav_pr = audio_prompt
    if not isinstance(wav_pr, torch.FloatTensor):
        wav_pr = torch.FloatTensor(wav_pr)
    if wav_pr.abs().max() > 1:
        wav_pr /= wav_pr.abs().max()
    if wav_pr.size(-1) == 2:
        wav_pr = wav_pr[:, 0]
    if wav_pr.ndim == 1:
        wav_pr = wav_pr.unsqueeze(0)
    assert wav_pr.ndim and wav_pr.size(0) == 1

    if transcript_content == "":
        text_pr, lang_pr = make_prompt(name, wav_pr, sr, save=False)
    else:
        lang_pr = langid.classify(str(transcript_content))[0]
        lang_token = lang2token[lang_pr]
        text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"
    # tokenize audio
    encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

    message = f"Detected language: {lang_pr}\n Detected text {text_pr}\n"

    # save as npz file
    np.savez(os.path.join(tempfile.gettempdir(), f"{name}.npz"),
             audio_tokens=audio_tokens, text_tokens=text_tokens, lang_code=lang2code[lang_pr])
    return message, os.path.join(tempfile.gettempdir(), f"{name}.npz")


def make_prompt(name, wav, sr, save=True):
    global whisper_model
    whisper_model.to(device)
    if not isinstance(wav, torch.FloatTensor):
        wav = torch.tensor(wav)
    if wav.abs().max() > 1:
        wav /= wav.abs().max()
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    assert wav.ndim and wav.size(0) == 1
    torchaudio.save(f"./prompts/{name}.wav", wav, sr)
    lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
    lang_token = lang2token[lang]
    text = lang_token + text + lang_token
    with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    if not save:
        os.remove(f"./prompts/{name}.wav")
        os.remove(f"./prompts/{name}.txt")

    whisper_model.cpu()
    torch.cuda.empty_cache()
    return text, lang

@torch.no_grad()
def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt, transcript_content):
    global model, text_collater, text_tokenizer, audio_tokenizer
    audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt
    sr, wav_pr = audio_prompt
    if not isinstance(wav_pr, torch.FloatTensor):
        wav_pr = torch.FloatTensor(wav_pr)
    if wav_pr.abs().max() > 1:
        wav_pr /= wav_pr.abs().max()
    if wav_pr.size(-1) == 2:
        wav_pr = wav_pr[:, 0]
    if wav_pr.ndim == 1:
        wav_pr = wav_pr.unsqueeze(0)
    assert wav_pr.ndim and wav_pr.size(0) == 1

    if transcript_content == "":
        text_pr, lang_pr = make_prompt('dummy', wav_pr, sr, save=False)
    else:
        lang_pr = langid.classify(str(transcript_content))[0]
        lang_token = lang2token[lang_pr]
        text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"

    if language == 'auto-detect':
        lang_token = lang2token[langid.classify(text)[0]]
    else:
        lang_token = langdropdown2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # onload model
    model.to(device)

    # tokenize audio
    encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
    audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

    # tokenize text
    logging.info(f"synthesize text: {text}")
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )

    enroll_x_lens = None
    if text_pr:
        text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
        text_prompts, enroll_x_lens = text_collater(
            [
                text_prompts
            ]
        )
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens
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
        best_of=5,
    )
    # Decode with Vocos
    frames = encoded_frames.permute(2,0,1)
    features = vocos.codes_to_features(frames)
    samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

    # offload model
    model.to('cpu')
    torch.cuda.empty_cache()

    message = f"text prompt: {text_pr}\nsythesized text: {text}"
    return message, (24000, samples.squeeze(0).cpu().numpy())

@torch.no_grad()
def infer_from_prompt(text, language, accent, preset_prompt, prompt_file):
    clear_prompts()
    model.to(device)
    # text to synthesize
    if language == 'auto-detect':
        lang_token = lang2token[langid.classify(text)[0]]
    else:
        lang_token = langdropdown2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # load prompt
    if prompt_file is not None:
        prompt_data = np.load(prompt_file.name)
    else:
        prompt_data = np.load(os.path.join("./presets/", f"{preset_prompt}.npz"))
    audio_prompts = prompt_data['audio_tokens']
    text_prompts = prompt_data['text_tokens']
    lang_pr = prompt_data['lang_code']
    lang_pr = code2lang[int(lang_pr)]

    # numpy to tensor
    audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
    text_prompts = torch.tensor(text_prompts).type(torch.int32)

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
        best_of=5,
    )
    # Decode with Vocos
    frames = encoded_frames.permute(2,0,1)
    features = vocos.codes_to_features(frames)
    samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

    model.to('cpu')
    torch.cuda.empty_cache()

    message = f"sythesized text: {text}"
    return message, (24000, samples.squeeze(0).cpu().numpy())


from utils.sentence_cutter import split_text_into_sentences
@torch.no_grad()
def infer_long_text(text, preset_prompt, prompt=None, language='auto', accent='no-accent'):
    """
    For long audio generation, two modes are available.
    fixed-prompt: This mode will keep using the same prompt the user has provided, and generate audio sentence by sentence.
    sliding-window: This mode will use the last sentence as the prompt for the next sentence, but has some concern on speaker maintenance.
    """
    mode = 'fixed-prompt'
    global model, audio_tokenizer, text_tokenizer, text_collater
    model.to(device)
    if (prompt is None or prompt == "") and preset_prompt == "":
        mode = 'sliding-window'  # If no prompt is given, use sliding-window mode
    sentences = split_text_into_sentences(text)
    # detect language
    if language == "auto-detect":
        language = langid.classify(text)[0]
    else:
        language = token2lang[langdropdown2token[language]]

    # if initial prompt is given, encode it
    if prompt is not None and prompt != "":
        # load prompt
        prompt_data = np.load(prompt.name)
        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]

        # numpy to tensor
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    elif preset_prompt is not None and preset_prompt != "":
        prompt_data = np.load(os.path.join("./presets/", f"{preset_prompt}.npz"))
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
        lang_pr = language if language != 'mix' else 'en'
    if mode == 'fixed-prompt':
        complete_tokens = torch.zeros([1, NUM_QUANTIZERS, 0]).type(torch.LongTensor).to(device)
        for text in sentences:
            text = text.replace("\n", "").strip(" ")
            if text == "":
                continue
            lang_token = lang2token[language]
            lang = token2lang[lang_token]
            text = lang_token + text + lang_token

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
                best_of=5,
            )
            complete_tokens = torch.cat([complete_tokens, encoded_frames.transpose(2, 1)], dim=-1)
        # Decode with Vocos
        frames = complete_tokens.permute(1, 0, 2)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

        model.to('cpu')
        message = f"Cut into {len(sentences)} sentences"
        return message, (24000, samples.squeeze(0).cpu().numpy())
    elif mode == "sliding-window":
        complete_tokens = torch.zeros([1, NUM_QUANTIZERS, 0]).type(torch.LongTensor).to(device)
        original_audio_prompts = audio_prompts
        original_text_prompts = text_prompts
        for text in sentences:
            text = text.replace("\n", "").strip(" ")
            if text == "":
                continue
            lang_token = lang2token[language]
            lang = token2lang[lang_token]
            text = lang_token + text + lang_token

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
                best_of=5,
            )
            complete_tokens = torch.cat([complete_tokens, encoded_frames.transpose(2, 1)], dim=-1)
            if torch.rand(1) < 1.0:
                audio_prompts = encoded_frames[:, :, -NUM_QUANTIZERS:]
                text_prompts = text_tokens[:, enroll_x_lens:]
            else:
                audio_prompts = original_audio_prompts
                text_prompts = original_text_prompts
        # Decode with Vocos
        frames = complete_tokens.permute(1, 0, 2)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

        model.to('cpu')
        message = f"Cut into {len(sentences)} sentences"
        return message, (24000, samples.squeeze(0).cpu().numpy())
    else:
        raise ValueError(f"No such mode {mode}")


def main():
    app = gr.Blocks(title="VALL-E X")
    with app:
        gr.Markdown(top_md)
        with gr.Tab("Infer from audio"):
            gr.Markdown(infer_from_audio_md)
            with gr.Row():
                with gr.Column():

                    textbox = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value="Welcome back, Master. What can I do for you today?", elem_id=f"tts-input")
                    language_dropdown = gr.Dropdown(choices=['auto-detect', 'English', '中文', '日本語'], value='auto-detect', label='language')
                    accent_dropdown = gr.Dropdown(choices=['no-accent', 'English', '中文', '日本語'], value='no-accent', label='accent')
                    textbox_transcript = gr.TextArea(label="Transcript",
                                          placeholder="Write transcript here. (leave empty to use whisper)",
                                          value="", elem_id=f"prompt-name")
                    upload_audio_prompt = gr.Audio(label='uploaded audio prompt', source='upload', interactive=True)
                    record_audio_prompt = gr.Audio(label='recorded audio prompt', source='microphone', interactive=True)
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(infer_from_audio,
                              inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, record_audio_prompt, textbox_transcript],
                              outputs=[text_output, audio_output])
                    textbox_mp = gr.TextArea(label="Prompt name",
                                          placeholder="Name your prompt here",
                                          value="prompt_1", elem_id=f"prompt-name")
                    btn_mp = gr.Button("Make prompt!")
                    prompt_output = gr.File(interactive=False)
                    btn_mp.click(make_npz_prompt,
                                inputs=[textbox_mp, upload_audio_prompt, record_audio_prompt, textbox_transcript],
                                outputs=[text_output, prompt_output])
            gr.Examples(examples=infer_from_audio_examples,
                        inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, record_audio_prompt, textbox_transcript],
                        outputs=[text_output, audio_output],
                        fn=infer_from_audio,
                        cache_examples=False,)
        with gr.Tab("Make prompt"):
            gr.Markdown(make_prompt_md)
            with gr.Row():
                with gr.Column():
                    textbox2 = gr.TextArea(label="Prompt name",
                                          placeholder="Name your prompt here",
                                          value="prompt_1", elem_id=f"prompt-name")
                    # 添加选择语言和输入台本的地方
                    textbox_transcript2 = gr.TextArea(label="Transcript",
                                          placeholder="Write transcript here. (leave empty to use whisper)",
                                          value="", elem_id=f"prompt-name")
                    upload_audio_prompt_2 = gr.Audio(label='uploaded audio prompt', source='upload', interactive=True)
                    record_audio_prompt_2 = gr.Audio(label='recorded audio prompt', source='microphone', interactive=True)
                with gr.Column():
                    text_output_2 = gr.Textbox(label="Message")
                    prompt_output_2 = gr.File(interactive=False)
                    btn_2 = gr.Button("Make!")
                    btn_2.click(make_npz_prompt,
                              inputs=[textbox2, upload_audio_prompt_2, record_audio_prompt_2, textbox_transcript2],
                              outputs=[text_output_2, prompt_output_2])
            gr.Examples(examples=make_npz_prompt_examples,
                        inputs=[textbox2, upload_audio_prompt_2, record_audio_prompt_2, textbox_transcript2],
                        outputs=[text_output_2, prompt_output_2],
                        fn=make_npz_prompt,
                        cache_examples=False,)
        with gr.Tab("Infer from prompt"):
            gr.Markdown(infer_from_prompt_md)
            with gr.Row():
                with gr.Column():
                    textbox_3 = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value="Welcome back, Master. What can I do for you today?", elem_id=f"tts-input")
                    language_dropdown_3 = gr.Dropdown(choices=['auto-detect', 'English', '中文', '日本語', 'Mix'], value='auto-detect',
                                                    label='language')
                    accent_dropdown_3 = gr.Dropdown(choices=['no-accent', 'English', '中文', '日本語'], value='no-accent',
                                                  label='accent')
                    preset_dropdown_3 = gr.Dropdown(choices=preset_list, value=None, label='Voice preset')
                    prompt_file = gr.File(file_count='single', file_types=['.npz'], interactive=True)
                with gr.Column():
                    text_output_3 = gr.Textbox(label="Message")
                    audio_output_3 = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn_3 = gr.Button("Generate!")
                    btn_3.click(infer_from_prompt,
                              inputs=[textbox_3, language_dropdown_3, accent_dropdown_3, preset_dropdown_3, prompt_file],
                              outputs=[text_output_3, audio_output_3])
            gr.Examples(examples=infer_from_prompt_examples,
                        inputs=[textbox_3, language_dropdown_3, accent_dropdown_3, preset_dropdown_3, prompt_file],
                        outputs=[text_output_3, audio_output_3],
                        fn=infer_from_prompt,
                        cache_examples=False,)
        with gr.Tab("Infer long text"):
            gr.Markdown("This is a long text generation demo. You can use this to generate long audio. ")
            with gr.Row():
                with gr.Column():
                    textbox_4 = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value=long_text_example, elem_id=f"tts-input")
                    language_dropdown_4 = gr.Dropdown(choices=['auto-detect', 'English', '中文', '日本語'], value='auto-detect',
                                                    label='language')
                    accent_dropdown_4 = gr.Dropdown(choices=['no-accent', 'English', '中文', '日本語'], value='no-accent',
                                                    label='accent')
                    preset_dropdown_4 = gr.Dropdown(choices=preset_list, value=None, label='Voice preset')
                    prompt_file_4 = gr.File(file_count='single', file_types=['.npz'], interactive=True)
                with gr.Column():
                    text_output_4 = gr.TextArea(label="Message")
                    audio_output_4 = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn_4 = gr.Button("Generate!")
                    btn_4.click(infer_long_text,
                              inputs=[textbox_4, preset_dropdown_4, prompt_file_4, language_dropdown_4, accent_dropdown_4],
                              outputs=[text_output_4, audio_output_4])

    webbrowser.open("http://127.0.0.1:7860")
    app.launch()

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
