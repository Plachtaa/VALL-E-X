# VALL-E X: 多语言文本到语音合成与语音克隆 🔊
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/qCBRmAnTxg)
<br>
[English](README.md) | 中文
<br>
微软[VALL-E X](https://arxiv.org/pdf/2303.03926) 零样本语音合成模型的开源实现.<br>
**预训练模型现已向公众开放，供研究或应用使用。**
![vallex-framework](/images/vallex_framework.jpg "VALL-E X framework")

VALL-E X 是一个强大而创新的多语言文本转语音（TTS）模型，最初由微软发布。虽然微软最初在他们的研究论文中提出了该概念，但并未发布任何代码或预训练模型。我们认识到了这项技术的潜力和价值，复现并训练了一个开源可用的VALL-E X模型。我们很乐意与社区分享我们的预训练模型，让每个人都能体验到次世代TTS的威力。 🎧
<br>
更多细节请查看 [model card](./model-card.md).

## 📖 目录
* [🚀 更新日志](#-更新日志)
* [📢 功能特点](#-功能特点)
* [💻 本地安装](#-本地安装)
* [🎧 在线Demo](#-在线Demo)
* [🐍 使用方法](#-Python中的使用方法)
* [❓ FAQ](#-faq)
* [🧠 TODO](#-todo)

## 🚀 Updates
**2023.09.10**
- 支持AR decoder的batch decoding以实现更稳定的生成结果

**2023.08.30**
- 将EnCodec解码器替换成了Vocos解码器，提升了音质。 (感谢[@v0xie](https://github.com/v0xie))

**2023.08.23**
- 加入了长文本生成功能

**2023.08.20**
- 加入了中文版README

**2023.08.14**
- 预训练模型权重已发布，从[这里](https://drive.google.com/file/d/10gdQWvP-K_e1undkvv0p2b7SU6I4Egyl/view?usp=sharing)下载。

## 💻 本地安装
### 使用pip安装，推荐使用Python 3.10，CUDA 11.7 ~ 12.0，PyTorch 2.0+
```commandline
git clone https://github.com/Plachtaa/VALL-E-X.git
cd VALL-E-X
pip install -r requirements.txt
```

> 注意：如果需要制作prompt，需要安装 ffmpeg 并将其所在文件夹加入到环境变量PATH中

第一次运行程序时，会自动下载相应的模型。如果下载失败并报错，请按照以下步骤手动下载模型。

（请注意目录和文件夹的大小写）

1.检查安装目录下是否存在`checkpoints`文件夹，如果没有，在安装目录下手动创建`checkpoints`文件夹（`./checkpoints/`）。

2.检查`checkpoints`文件夹中是否有`vallex-checkpoint.pt`文件。如果没有，请从[这里](https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt)
手动下载`vallex-checkpoint.pt`文件并放到`checkpoints`文件夹里。

3.检查安装目录下是否存在`whisper`文件夹，如果没有，在安装目录下手动创建`whisper`文件夹（`./whisper/`）。

4.检查`whisper`文件夹中是否有`medium.pt`文件。如果没有，请从[这里](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)
手动下载`medium.pt`文件并放到`whisper`文件夹里。

##  🎧 在线Demo
如果你不想在本地安装，你可以在线体验VALL-E X的功能，点击下面的任意一个链接即可开始体验。
<br>
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/Plachta/VALL-E-X)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yyD_sz531QntLKowMHo-XxorsFBCfKul?usp=sharing)


## 📢 功能特点

VALL-E X 配备有一系列尖端功能：

1. **多语言 TTS**: 可使用三种语言 - 英语、中文和日语 - 进行自然、富有表现力的语音合成。

2. **零样本语音克隆**: 仅需录制任意说话人的短短的 3~10 秒录音，VALL-E X 就能生成个性化、高质量的语音，完美还原他们的声音。

<details>
  <summary><h5>查看示例</h5></summary>

[prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/a7baa51d-a53a-41cc-a03d-6970f25fcca7)


[output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/b895601a-d126-4138-beff-061aabdc7985)

</details>

3. **语音情感控制**: VALL-E X 可以合成与给定说话人录音相同情感的语音，为音频增添更多表现力。

<details>
  <summary><h5>查看示例</h5></summary>

https://github.com/Plachtaa/VALL-E-X/assets/112609742/56fa9988-925e-4757-82c5-83ecb0df6266


https://github.com/Plachtaa/VALL-E-X/assets/112609742/699c47a3-d502-4801-8364-bd89bcc0b8f1

</details>

4. **零样本跨语言语音合成**: VALL-E X 可以合成与给定说话人母语不同的另一种语言，在不影响口音和流利度的同时，保留该说话人的音色与情感。以下是一个使用日语母语者进行英文与中文合成的样例： 🇯🇵 🗣

<details>
  <summary><h5>查看示例</h5></summary>

[jp-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/ea6e2ee4-139a-41b4-837e-0bd04dda6e19)


[en-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/db8f9782-923f-425e-ba94-e8c1bd48f207)


[zh-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/15829d79-e448-44d3-8965-fafa7a3f8c28)

</details>

5. **口音控制**: VALL-E X 允许您控制所合成音频的口音，比如说中文带英语口音或反之。 🇨🇳 💬

<details>
  <summary><h5>查看示例</h5></summary>

[en-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/f688d7f6-70ef-46ec-b1cc-355c31e78b3b)


[zh-accent-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/be59c7ca-b45b-44ca-a30d-4d800c950ccc)


[en-accent-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/8b4f4f9b-f299-4ea4-a548-137437b71738)

</details>

6. **声学环境保留**: 当给定说话人的录音在不同的声学环境下录制时，VALL-E X 可以保留该声学环境，使合成语音听起来更加自然。

<details>
  <summary><h5>查看示例</h5></summary>

[noise-prompt.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/68986d88-abd0-4d1d-96e4-4f893eb9259e)


[noise-output.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/96c4c612-4516-4683-8804-501b70938608)

</details>


你可以访问我们的[demo页面](https://plachtaa.github.io/) 来浏览更多示例!

## 💻 Python中的使用方法

<details open>
  <summary><h3>🪑 基本使用</h3></summary>

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
  <summary><h3>🌎 多语言</h3></summary>
<br>
该VALL-E X实现支持三种语言：英语、中文和日语。您可以通过设置`language`参数来指定语言。默认情况下，该模型将自动检测语言。
<br>

```python

text_prompt = """
    チュソクは私のお気に入りの祭りです。 私は数日間休んで、友人や家族との時間を過ごすことができます。
"""
audio_array = generate_audio(text_prompt)
```

[vallex_japanese.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/ee57a688-3e83-4be5-b0fe-019d16eec51c)

*注意：即使在一句话中混合多种语言的情况下，VALL-E X也能完美地控制口音，但是您需要手动标记各个句子对应的语言以便于我们的G2P工具识别它们。*
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
<summary><h3>📼 预设音色</h3></summary>
  
我们提供十几种说话人音色可直接VALL-E X使用! 在[这里](/presets)浏览所有可用音色。

> VALL-E X 尝试匹配给定预设音色的音调、音高、情感和韵律。该模型还尝试保留音乐、环境噪声等。
```python
text_prompt = """
I am an innocent boy with a smoky voice. It is a great honor for me to speak at the United Nations today.
"""
audio_array = generate_audio(text_prompt, prompt="dingzhen")
```

[smoky.webm](https://github.com/Plachtaa/VALL-E-X/assets/112609742/d3f55732-b1cd-420f-87d6-eab60db14dc5)

</details>

<details open>
<summary><h3>🎙声音克隆</h3></summary>
  
VALL-E X 支持声音克隆！你可以使用任何人，角色，甚至是你自己的声音，来制作一个音频提示。在你使用该音频提示时，VALL-E X 将会使用与其相似的声音来合成文本。
<br>
你需要提供一段3~10秒长的语音，以及该语音对应的文本，来制作音频提示。你也可以将文本留空，让[Whisper](https://github.com/openai/whisper)模型为你生成文本。
> VALL-E X 尝试匹配给定音频提示的音调、音高、情感和韵律。该模型还尝试保留音乐、环境噪声等。

```python
from utils.prompt_making import make_prompt

### Use given transcript
make_prompt(name="paimon", audio_prompt_path="paimon_prompt.wav",
                transcript="Just, what was that? Paimon thought we were gonna get eaten.")

### Alternatively, use whisper
make_prompt(name="paimon", audio_prompt_path="paimon_prompt.wav")
```
来尝试一下刚刚做好的音频提示吧！
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
<summary><h3>🎢用户界面</h3></summary>

如果你不擅长代码，我们还为VALL-E X创建了一个用户友好的图形界面。它可以让您轻松地与模型进行交互，使语音克隆和多语言语音合成变得轻而易举。
<br>
使用以下命令启动用户界面：
```commandline
python -X utf8 launch-ui.py
```
</details>

## 🛠️ 硬件要求及推理速度

VALL-E X 可以在CPU或GPU上运行 (`pytorch 2.0+`, CUDA 11.7 ~ CUDA 12.0).

若使用GPU运行，你需要至少6GB的显存。

## ⚙️ Details

VALL-E X 与 [Bark](https://github.com/suno-ai/bark), [VALL-E](https://arxiv.org/abs/2301.02111) and [AudioLM](https://arxiv.org/abs/2209.03143)类似, 使用GPT风格的模型以自回归方式预测量化音频token，并由[EnCodec](https://github.com/facebookresearch/encodec)解码.
<br>
与 [Bark](https://github.com/suno-ai/bark) 相比:
- ✔ **轻量**: 3️⃣ ✖ 更小,
- ✔ **快速**: 4️⃣ ✖ 更快, 
- ✔ **中文&日文的更高质量**
- ✔ **跨语言合成时没有外国口音**
- ✔ **开放且易于操作的声音克隆**
- ❌ **支持的语言较少**
- ❌ **没有用于合成音乐及特殊音效的token**

### 支持的语言

| 语言      | 状态 |
|---------| :---: |
| 英语 (en) | ✅ |
| 日语 (ja) | ✅ |
| 中文 (zh) | ✅ |

## ❓ FAQ

#### 在哪里可以下载checkpoint?
* 当您第一次运行程序时,我们使用`wget`将模型下载到`./checkpoints/`目录里。
* 如果第一次运行时下载失败，请从[这里](https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt)手动下载模型，并将文件放在`./checkpoints/`里。

#### 需要多少显存?
* 6GB 显存(GPU VRAM) - 几乎所有NVIDIA GPU都满足要求.

#### 为什么模型无法生成长文本?
* 当序列长度增加时，Transformer的计算复杂度呈二次方增长。因此，所有训练音频都保持在22秒以下。请确保音频提示（audio prompt）和生成的音频的总长度小于22秒以确保可接受的性能。

#### 更多...

## 🧠 待办事项
- [x] 添加中文 README
- [x] 长文本生成
- [x] 用Vocos解码器替换Encodec解码器
- [ ] 微调以实现更好的语音自适应
- [ ] 给非python用户的`.bat`脚本
- [ ] 更多...

## 🙏 感谢
- [VALL-E X paper](https://arxiv.org/pdf/2303.03926) for the brilliant idea
- [lifeiteng's vall-e](https://github.com/lifeiteng/vall-e) for related training code
- [bark](https://github.com/suno-ai/bark) for the amazing pioneering work in neuro-codec TTS model

## ⭐️ 表示出你的支持

如果您觉得VALL-E X有趣且有用，请在GitHub上给我们一颗星！ ⭐️ 它鼓励我们不断改进模型并添加令人兴奋的功能。

## 📜 License

VALL-E X 使用 [MIT License](./LICENSE).

---

有问题或需要帮助？ 可以随便 [open an issue](https://github.com/Plachtaa/VALL-E-X/issues/new) 或加入我们的 [Discord](https://discord.gg/qCBRmAnTxg)

Happy voice cloning! 🎤