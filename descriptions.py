top_md = """
# VALL-E X  
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

long_text_md = """
Very long text is chunked into several sentences, and each sentence is synthesized separately.<br>
Please make a prompt or use a preset prompt to infer long text.
"""

long_text_example = "Just a few years ago, there were no legions of deep learning scientists developing intelligent products and services at major companies and startups. When we entered the field, machine learning did not command headlines in daily newspapers. Our parents had no idea what machine learning was, let alone why we might prefer it to a career in medicine or law. Machine learning was a blue skies academic discipline whose industrial significance was limited to a narrow set of real-world applications, including speech recognition and computer vision. Moreover, many of these applications required so much domain knowledge that they were often regarded as entirely separate areas for which machine learning was one small component. At that time, neural networks—the predecessors of the deep learning methods that we focus on in this book—were generally regarded as outmoded."