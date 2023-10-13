from setuptools import find_packages, setup

setup(
    name="VALL-E-X",
    packages=find_packages(exclude=[]),
    description="An open source implementation of Microsoft's VALL-E X zero-shot TTS",
    author="Plachtaa",
    long_description_content_type="text/markdown",
    keywords=[
        "artificial intelligence",
        "deep learning",
    ],
    dependency_links=[
        'https://download.pytorch.org/whl/cu121'
    ],
    install_requires=[
        "soundfile",
        "numpy",
        "torch",
        "torchvision",
        "torchaudio",
        "tokenizers",
        "encodec",
        "langid",
        "wget",
        "unidecode",
        "pyopenjtalk-prebuilt",
        "pypinyin",
        "inflect",
        "cn2an",
        "jieba",
        "eng_to_ipa",
        "openai-whisper",
        "matplotlib",
        "gradio",
        "nltk",
        "sudachipy",
        "sudachidict_core",
        "vocos",
        "lhotse",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
