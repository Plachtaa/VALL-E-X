top_md = """
# VALL-E X  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yyD_sz531QntLKowMHo-XxorsFBCfKul?usp=sharing)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Plachtaa/vallex-webui)
Unofficial implementation of Microsoft's [VALL-E X](https://arxiv.org/pdf/2303.03926).<br>
VALL-E X can synthesize high-quality personalized speech with only a 3-second enrolled recording of 
an unseen speaker as an acoustic prompt, even in another language for a monolingual speaker.<br>
This implementation supports zero-shot, mono-lingual/cross-lingual text-to-speech functionality of three languages (English, Chinese, Japanese)<br>  
See this [demo](https://plachtaa.github.io/) page for more details.
"""

infer_from_audio_md = """
Upload a speech of 3~10 seconds as the audio prompt and type in the text you'd like to synthesize.<br>
The model will synthesize speech of given text with the same voice of your audio prompt.<br>
The model also tends to preserve the emotion & acoustic environment of your given speech.<br>
For faster inference, please use **"Make prompt"** to get a `.npz` file as the encoded audio prompt, and use it by **"Infer from prompt"**
"""

make_prompt_md = """
Upload a speech of 3~10 seconds as the audio prompt.<br>
Get a `.npz` file as the encoded audio prompt. Use it by **"Infer with prompt"**
"""

infer_from_prompt_md = """
Faster than **"Infer from audio"**.<br>
You need to **"Make prompt"** first, and upload the encoded prompt (a `.npz` file)
"""