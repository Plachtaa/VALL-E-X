# VALL-E X: Multilingual Text-to-Speech Synthesis and Voice Cloning 🔊
An open source implementation of Microsoft's [VALL-E X](https://arxiv.org/pdf/2303.03926) zero-shot TTS model.<br>
**Trained model will be released after this repository is fully ready.**

![vallex-framework](/images/vallex_framework.jpg "VALL-E X framework")

## ⭐️ Welcome to VALL-E X! ⭐️

VALL-E X is an amazing multilingual text-to-speech (TTS) model inspired by Microsoft's groundbreaking research. While Microsoft initially proposed the concept in their research paper, they did not release any code or pretrained models. Recognizing the potential and value of this technology, our team took on the challenge to reproduce the results and train our own model. We are excited to share our trained VALL-E X model with the community, allowing everyone to experience the power of personalized speech synthesis and voice cloning! 🎧
<br>
<br>
More details about the model are presented in [model card](./model-card.md).
## 💻 Installation
### Install with pip
```commandline
https://github.com/Plachtaa/VALL-E-X.git
cd VALL-E-X
pip install --no-error-on-external -r requirements.txt
```
### ❗❗❗ Special Notes ❗❗❗
Japanese g2p tool `pyopenjtalk` may fail to build during installation, you may ignore it if you don't require Japanese TTS functionality.
We are currently searching for more stable substitution.
##  🎧 Demos
Not ready to set up the environment on your local machine just yet? No problem! We've got you covered with our online demos. You can try out VALL-E X directly on Hugging Face or Google Colab, experiencing the model's capabilities hassle-free!
<br>
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()


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
from IPython.display import Audio

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
Hey, Traveler, Listen to this, This machine has taken my voice, and now it can talk just like me!
"""
audio_array = generate_audio(text_prompt, prompt="paimon")

# save audio to disk
write_wav("paimon_cloned.wav", SAMPLE_RATE, audio_array)

# play text in notebook
Audio(audio_array, rate=SAMPLE_RATE)
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
python launch-ui.py
```
</details>

## 🙌 Contribute

We welcome contributions from the community to make VALL-E X even better! If you have ideas, bug fixes, or want to add more languages, emotions, or accents to the model, feel free to submit a pull request. Together, we can take VALL-E X to new heights!

## ⭐️ Show Your Support

If you find VALL-E X interesting and useful, give us a star on GitHub! ⭐️ It encourages us to keep improving the model and adding exciting features.

## 📜 License

VALL-E X is licensed under the [MIT License](./LICENSE).

---

Let your imagination run wild with VALL-E X, and enjoy the fantastic world of multilingual text-to-speech synthesis and voice cloning! 🌈 🎶

Have questions or need assistance? Feel free to open an issue or join our community (not set yet lol)

Happy voice cloning! 🎤
