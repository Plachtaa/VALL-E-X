NUM_LAYERS = 12
NUM_HEAD = 16
N_DIM = 1024
PREFIX_MODE = 1
NUM_QUANTIZERS = 8
SAMPLE_RATE = 24000
<<<<<<< HEAD
#updated
=======

>>>>>>> master
lang2token = {
    'zh': "[ZH]",
    'ja': "[JA]",
    "en": "[EN]",
<<<<<<< HEAD
=======
    "ar": "[AR]",
>>>>>>> master
    'mix': "",
}

lang2code = {
    'zh': 0,
    'ja': 1,
    "en": 2,
<<<<<<< HEAD
=======
    "ar": 3,
>>>>>>> master
}

token2lang = {
    '[ZH]': "zh",
    '[JA]': "ja",
    "[EN]": "en",
<<<<<<< HEAD
=======
    "[AR]": "ar",
>>>>>>> master
    "": "mix"
}

code2lang = {
    0: 'zh',
    1: 'ja',
    2: "en",
<<<<<<< HEAD
=======
    3: "ar",
>>>>>>> master
}

langdropdown2token = {
    'English': "[EN]",
    '中文': "[ZH]",
    '日本語': "[JA]",
<<<<<<< HEAD
=======
    'عربي':"[AR]",
>>>>>>> master
    'Mix': "",
}