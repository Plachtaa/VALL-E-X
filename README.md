# VALL-E X: Multilingual Text-to-Speech Synthesis and Voice Cloning 🔊
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/qCBRmAnTxg)
<br>
English | [中文](README-ZH.md)
<br>
An open source implementation of Microsoft's [VALL-E X](https://arxiv.org/pdf/2303.03926) zero-shot TTS model.<br>
**We release our trained model to the public for research or application usage.**

![vallex-framework](/images/vallex_framework.jpg "VALL-E X framework")

VALL-E X is an amazing multilingual text-to-speech (TTS) model proposed by Microsoft. While Microsoft initially publish in their research paper, they did not release any code or pretrained models. Recognizing the potential and value of this technology, our team took on the challenge to reproduce the results and train our own model. We are glad to share our trained VALL-E X model with the community, allowing everyone to experience the power next-generation TTS! 🎧
<br>
<br>
More details about the model are presented in [model card](./model-card.md).

## 📖 Quick Index
* [🚀 Updates](#-updates)
* [📢 Features](#-features)
* [💻 Installation](#-installation)
* [🎧 Demos](#-demos)
* [🐍 Usage](#-usage-in-python)
* [❓ FAQ](#-faq)
* [🧠 TODO](#-todo)

## 🚀 Updates
**2023.09.10**
- Added AR decoder batch decoding for more stable generation result.

**2023.08.30**
- Replaced EnCodec decoder with Vocos decoder, improved audio quality. (Thanks to [@v0xie](https://github.com/v0xie))

**2023.08.23**
- Added long text generation.

**2023.08.20**
- Added [Chinese README](README-ZH.md).

**2023.08.14**
- Pretrained VALL-E X checkpoint is now released. Download it [here](https://drive.google.com/file/d/10gdQWvP-K_e1undkvv0p2b7SU6I4Egyl/view?usp=sharing)

## 💻 Installation
### Install with pip, recommended with Python 3.10, CUDA 11.7 ~ 12.0, PyTorch 2.0+
```commandline
git clone https://github.com/Plachtaa/VALL-E-X.git
cd VALL-E-X
pip install -r requirements.txt
```

> Note: If you want to make prompt, you need to install ffmpeg and add its folder to the environment variable PATH.

When you run the program for the first time, it will automatically download the corresponding model. 

If the download fails and reports an error, please follow the steps below to manually download the model.

(Please pay attention to the capitalization of folders)

1. Check whether there is a `checkpoints` folder in the installation directory. 
If not, manually create a `checkpoints` folder (`./checkpoints/`) in the installation directory.

2. Check whether there is a `vallex-checkpoint.pt` file in the `checkpoints` folder. 
If not, please manually download the `vallex-checkpoint.pt` file from [here](https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt) and put it in the `checkpoints` folder.

3. Check whether there is a `whisper` folder in the installation directory. 
If not, manually create a `whisper` folder (`./whisper/`) in the installation directory.

4. Check whether there is a `medium.pt` file in the `whisper` folder. 
If not, please manually download the `medium.pt` file from [here](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt) and put it in the `whisper` folder.

##  🎧 Demos
Not ready to set up the environment on your local machine just yet? No problem! We've got you covered with our online demos. You can try out VALL-E X directly on Hugging Face or Google Colab, experiencing the model's capabilities hassle-free!
<br>
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/Plachta/VALL-E-X)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yyD_sz531QntLKowMHo-XxorsFBCfKul?usp=sharing)


## 📢 Features

VALL-E X comes packed with cutting-edge functionalities:

1. **Multilingual TTS**: Speak in three languages - English, Chinese, and Japanese - with natural and expressive speech synthesis.

2. **Zero-shot Voice Cloning**: Enroll a short 3~10 seconds recording of an unseen speaker, and watch VALL-E X create personalized, high-quality speech that sounds just like them!

<details>
  <summary><h5>see example</h5></summary>

[prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/a7baa51d-a53a-41cc-a03d-6970f25fcca7)


[output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/b895601a-d126-4138-beff-061aabdc7985)

</details>

3. **Speech Emotion Control**: Experience the power of emotions! VALL-E X can synthesize speech with the same emotion as the acoustic prompt provided, adding an extra layer of expressiveness to your audio.

<details>
  <summary><h5>see example</h5></summary>

https://github.com/Plachtaa/VALL-E-X/assets/112609742/56fa9988-925e-4757-82c5-83ecb0df6266


https://github.com/Plachtaa/VALL-E-X/assets/112609742/699c47a3-d502-4801-8364-bd89bcc0b8f1

</details>

4. **Zero-shot Cross-Lingual Speech Synthesis**: Take monolingual speakers on a linguistic journey! VALL-E X can produce personalized speech in another language without compromising on fluency or accent. Below is a Japanese speaker talk in Chinese & English. 🇯🇵 🗣

<details>
  <summary><h5>see example</h5></summary>

[jp-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/ea6e2ee4-139a-41b4-837e-0bd04dda6e19)


[en-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/db8f9782-923f-425e-ba94-e8c1bd48f207)


[zh-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/15829d79-e448-44d3-8965-fafa7a3f8c28)

</details>

5. **Accent Control**: Get creative with accents! VALL-E X allows you to experiment with different accents, like speaking Chinese with an English accent or vice versa. 🇨🇳 💬

<details>
  <summary><h5>see example</h5></summary>

[en-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/f688d7f6-70ef-46ec-b1cc-355c31e78b3b)


[zh-accent-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/be59c7ca-b45b-44ca-a30d-4d800c950ccc)


[en-accent-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/8b4f4f9b-f299-4ea4-a548-137437b71738)

</details>

6. **Acoustic Environment Maintenance**: No need for perfectly clean audio prompts! VALL-E X adapts to the acoustic environment of the input, making speech generation feel natural and immersive.

<details>
  <summary><h5>see example</h5></summary>

[noise-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/68986d88-abd0-4d1d-96e4-4f893eb9259e)


[noise-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/96c4c612-4516-4683-8804-501b70938608)

</details>


Explore our [demo page](https://plachtaa.github.io/) for a lot more examples!

## 🐍 Usage in Python

<details open>
  <summary><h3>🪑 Basics</h3></summary>

```python
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
Hello, my name is Nose. And uh, and I like hamburger. Hahaha... But I also have other interests such as playing tactic toast.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("vallex_generation.wav", SAMPLE_RATE, audio_array)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
```

[hamburger.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/578d7bbe-cda9-483e-898c-29646edc8f2e)

</details>

<details open>
  <summary><h3>🌎 Foreign Language</h3></summary>
<br>
This VALL-E X implementation also supports Chinese and Japanese. All three languages have equally awesome performance!
<br>

```python

text_prompt = """
    チュソクは私のお気に入りの祭りです。 私は数日間休んで、友人や家族との時間を過ごすことができます。
"""
audio_array = generate_audio(text_prompt)
```

[vallex_japanese.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/ee57a688-3e83-4be5-b0fe-019d16eec51c)

*Note: VALL-E X controls accent perfectly even when synthesizing code-switch text. However, you need to manually denote language of respective sentences (since our g2p tool is rule-base)*
```python
text_prompt = """
    [EN]The Thirty Years' War was a devastating conflict that had a profound impact on Europe.[EN]
    [ZH]这是历史的开始。 如果您想听更多，请继续。[ZH]
"""
audio_array = generate_audio(text_prompt, language='mix')
```

[vallex_codeswitch.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/d8667abf-bd08-499f-a383-a861d852f98a)

</details>

<details open>
<summary><h3>📼 Voice Presets</h3></summary>
  
VALL-E X provides tens of speaker voices which you can directly used for inference! Browse all voices in the [code](/presets)

> VALL-E X tries to match the tone, pitch, emotion and prosody of a given preset. The model also attempts to preserve music, ambient noise, etc.

```python
text_prompt = """
I am an innocent boy with a smoky voice. It is a great honor for me to speak at the United Nations today.
"""
audio_array = generate_audio(text_prompt, prompt="dingzhen")
```

[smoky.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/d3f55732-b1cd-420f-87d6-eab60db14dc5)

</details>

<details open>
<summary><h3>🎙Voice Cloning</h3></summary>
  
VALL-E X supports voice cloning! You can make a voice prompt with any person, character or even your own voice, and use it like other voice presets.<br>
To make a voice prompt, you need to provide a speech of 3~10 seconds long, as well as the transcript of the speech. 
You can also leave the transcript blank to let the [Whisper](https://github.com/openai/whisper) model to generate the transcript.
> VALL-E X tries to match the tone, pitch, emotion and prosody of a given prompt. The model also attempts to preserve music, ambient noise, etc.

```python
from utils.prompt_making import make_prompt

### Use given transcript
make_prompt(name="paimon", audio_prompt_path="paimon_prompt.wav",
                transcript="Just, what was that? Paimon thought we were gonna get eaten.")

### Alternatively, use whisper
make_prompt(name="paimon", audio_prompt_path="paimon_prompt.wav")
```
Now let's try out the prompt we've just made!
```python
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models()

text_prompt = """
Hey, Traveler, Listen to this, This machine has taken my voice, and now it can talk just like me!
"""
audio_array = generate_audio(text_prompt, prompt="paimon")

write_wav("paimon_cloned.wav", SAMPLE_RATE, audio_array)

```

[paimon_prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/e7922859-9d12-4e2a-8651-e156e4280311)


[paimon_cloned.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/60d3b7e9-5ead-4024-b499-a897ce5f3d5e)


</details>


<details open>
<summary><h3>🎢User Interface</h3></summary>

Not comfortable with codes? No problem! We've also created a user-friendly graphical interface for VALL-E X. It allows you to interact with the model effortlessly, making voice cloning and multilingual speech synthesis a breeze.
<br>
You can launch the UI by the following command:
```commandline
python -X utf8 launch-ui.py
```
</details>

## 🛠️ Hardware and Inference Speed

VALL-E X works well on both CPU and GPU (`pytorch 2.0+`, CUDA 11.7 and CUDA 12.0).

A GPU VRAM of 6GB is enough for running VALL-E X without offloading.

## ⚙️ Details

VALL-E X is similar to [Bark](https://github.com/suno-ai/bark), [VALL-E](https://arxiv.org/abs/2301.02111) and [AudioLM](https://arxiv.org/abs/2209.03143), which generates audio in GPT-style by predicting audio tokens quantized by [EnCodec](https://github.com/facebookresearch/encodec).
<br>
Comparing to [Bark](https://github.com/suno-ai/bark):
- ✔ **Light-weighted**: 3️⃣ ✖ smaller,
- ✔ **Efficient**: 4️⃣ ✖ faster, 
- ✔ **Better quality on Chinese & Japanese**
- ✔ **Cross-lingual speech without foreign accent**
- ✔ **Easy voice-cloning**
- ❌ **Less languages**
- ❌ **No special tokens for music / sound effects**

### Supported Languages

| Language | Status |
| --- | :---: |
| English (en) | ✅ |
| Japanese (ja) | ✅ |
| Chinese, simplified (zh) | ✅ |

## ❓ FAQ

#### Where can I download the model checkpoint?
* We use `wget` to download the model to directory `./checkpoints/` when you run the program for the first time.
* If the download fails on the first run, please manually download from [this link](https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt), and put the file under directory `./checkpoints/`.

#### How much VRAM do I need?
* 6GB GPU VRAM - Almost all NVIDIA GPUs satisfy the requirement.

#### Why the model fails to generate long text?
* Transformer's computation complexity increases quadratically while the sequence length increases. Hence, all training 
are kept under 22 seconds. Please make sure the total length of audio prompt and generated audio is less than 22 seconds 
to ensure acceptable performance. 


#### MORE TO BE ADDED...

## 🧠 TODO
- [x] Add Chinese README
- [x] Long text generation
- [x] Replace Encodec decoder with Vocos decoder
- [ ] Fine-tuning for better voice adaptation
- [ ] `.bat` scripts for non-python users
- [ ] To be added...

## 🙏 Appreciation
- [VALL-E X paper](https://arxiv.org/pdf/2303.03926) for the brilliant idea
- [lifeiteng's vall-e](https://github.com/lifeiteng/vall-e) for related training code
- [bark](https://github.com/suno-ai/bark) for the amazing pioneering work in neuro-codec TTS model

## ⭐️ Show Your Support

If you find VALL-E X interesting and useful, give us a star on GitHub! ⭐️ It encourages us to keep improving the model and adding exciting features.

## 📜 License

VALL-E X is licensed under the [MIT License](./LICENSE).

---

Have questions or need assistance? Feel free to [open an issue](https://github.com/Plachtaa/VALL-E-X/issues/new) or join our [Discord](https://discord.gg/qCBRmAnTxg)

Happy voice cloning! 🎤
