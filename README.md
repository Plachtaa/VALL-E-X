# VALL-E X: Multilingual Text-to-Speech Synthesis and Voice Cloning 🔊
An open source implementation of Microsoft's [VALL-E X](https://arxiv.org/pdf/2303.03926) zero-shot TTS model.<br>
**Trained model will be released after this repository is fully ready.**

![vallex-framework](/images/vallex_framework.jpg "VALL-E X framework")

## ⭐️ Welcome to VALL-E X! ⭐️

VALL-E X is an amazing multilingual text-to-speech (TTS) model inspired by Microsoft's groundbreaking research. While Microsoft initially proposed the concept in their research paper, they did not release any code or pretrained models. Recognizing the potential and value of this technology, our team took on the challenge to reproduce the results and train our own model. We are excited to share our trained VALL-E X model with the community, allowing everyone to experience the power of personalized speech synthesis and voice cloning! 🎧

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

[multilingual-tts.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/11b5a789-45ad-4e22-9450-acbe80a3a9d7)

2. **Zero-shot Voice Cloning**: Enroll a short 3~10 seconds recording of an unseen speaker, and watch VALL-E X create personalized, high-quality speech that sounds just like them!

[prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/59bcf1a2-1b21-42ec-91cc-9b0b9925df08)


[output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/517c77bd-55cd-4ff8-b1c0-3aa0e963a1b2)

3. **Speech Emotion Control**: Experience the power of emotions! VALL-E X can synthesize speech with the same emotion as the acoustic prompt provided, adding an extra layer of expressiveness to your audio.


https://github.com/Plachtaa/VALL-E-X/assets/112609742/56fa9988-925e-4757-82c5-83ecb0df6266


https://github.com/Plachtaa/VALL-E-X/assets/112609742/699c47a3-d502-4801-8364-bd89bcc0b8f1



4. **Zero-shot Cross-Lingual Speech Synthesis**: Take monolingual speakers on a linguistic journey! VALL-E X can produce personalized speech in another language without compromising on fluency or accent. Below is a Japanese speaker talk in Chinese & English. 🇯🇵 🗣

[jp-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/ea6e2ee4-139a-41b4-837e-0bd04dda6e19)


[en-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/db8f9782-923f-425e-ba94-e8c1bd48f207)


[zh-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/15829d79-e448-44d3-8965-fafa7a3f8c28)

5. **Accent Control**: Get creative with accents! VALL-E X allows you to experiment with different accents, like speaking Chinese with an English accent or vice versa. 🇨🇳 💬

[en-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/f688d7f6-70ef-46ec-b1cc-355c31e78b3b)


[zh-accent-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/be59c7ca-b45b-44ca-a30d-4d800c950ccc)


[en-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/81d7c190-71cc-4ab4-8b33-ccccf0de8d06)

6. **Acoustic Environment Maintenance**: No need for perfectly clean audio prompts! VALL-E X adapts to the acoustic environment of the input, making speech generation feel natural and immersive.

[noise-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/c45a0fc4-bb04-4bfb-bc3f-5ba7c275f2ce)


[noise-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/1b647ca8-805c-4582-b268-6fa042867412)

Explore our [demo page](https://plachtaa.github.io/) for a lot more examples!

## 🐍 Usage in Python

### API (TODO)

Integrate VALL-E X into your projects with ease using our simple API. The API provides access to all of the model's fantastic functionalities, empowering you to build exciting applications with personalized speech synthesis capabilities.

### UI

Not comfortable with APIs? No problem! We've also created a user-friendly graphical interface for VALL-E X. It allows you to interact with the model effortlessly, making voice cloning and multilingual speech synthesis a breeze.

## 🚀 Getting Started

Follow these simple steps to get started with VALL-E X:

1. Clone this repository to your local machine.
2. (To be completed)

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
