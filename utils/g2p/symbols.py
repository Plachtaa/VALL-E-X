'''
Defines the set of symbols used in text input to the model.
'''

# japanese_cleaners
# _pad        = '_'
# _punctuation = ',.!?-'
# _letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧ↓↑ '


'''# japanese_cleaners2
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧʦ↓↑ '
'''


'''# korean_cleaners
_pad        = '_'
_punctuation = ',.!?…~'
_letters = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ '
'''

'''# chinese_cleaners
_pad        = '_'
_punctuation = '，。！？—…'
_letters = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ '
'''

# # zh_ja_mixture_cleaners
# _pad        = '_'
# _punctuation = ',.!?-~…'
# _letters = 'AEINOQUabdefghijklmnoprstuvwyzʃʧʦɯɹəɥ⁼ʰ`→↓↑ '


'''# sanskrit_cleaners
_pad        = '_'
_punctuation = '।'
_letters = 'ँंःअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसहऽािीुूृॄेैोौ्ॠॢ '
'''

'''# cjks_cleaners
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzʃʧʥʦɯɹəɥçɸɾβŋɦː⁼ʰ`^#*=→↓↑ '
'''

'''# thai_cleaners
_pad        = '_'
_punctuation = '.!? '
_letters = 'กขฃคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์'
'''

# # cjke_cleaners2
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '


'''# shanghainese_cleaners
_pad        = '_'
_punctuation = ',.!?…'
_letters = 'abdfghiklmnopstuvyzøŋȵɑɔɕəɤɦɪɿʑʔʰ̩̃ᴀᴇ15678 '
'''

'''# chinese_dialect_cleaners
_pad        = '_'
_punctuation = ',.!?~…─'
_letters = '#Nabdefghijklmnoprstuvwxyzæçøŋœȵɐɑɒɓɔɕɗɘəɚɛɜɣɤɦɪɭɯɵɷɸɻɾɿʂʅʊʋʌʏʑʔʦʮʰʷˀː˥˦˧˨˩̥̩̃̚ᴀᴇ↑↓∅ⱼ '
'''
''' # arabic cleaners

PADDING_TOKEN = '_pad_'
EOS_TOKEN = '_eos_'
DOUBLING_TOKEN = '_dbl_'
SEPARATOR_TOKEN = '_+_'

EOS_TOKENS = [SEPARATOR_TOKEN, EOS_TOKEN]

symbols = [
    # special tokens
    PADDING_TOKEN,  # padding
    EOS_TOKEN,  # eos-token
    '_sil_',  # silence
    DOUBLING_TOKEN,  # doubling
    SEPARATOR_TOKEN,  # word separator
    # consonants
    '<',  # hamza
    'b',  # baa'
    't',  # taa'
    '^',  # thaa'
    'j',  # jiim
    'H',  # Haa'
    'x',  # xaa'
    'd',  # daal
    '*',  # dhaal
    'r',  # raa'
    'z',  # zaay
    's',  # siin
    '$',  # shiin
    'S',  # Saad
    'D',  # Daad
    'T',  # Taa'
    'Z',  # Zhaa'
    'E',  # 3ayn
    'g',  # ghain
    'f',  # faa'
    'q',  # qaaf
    'k',  # kaaf
    'l',  # laam
    'm',  # miim
    'n',  # nuun
    'h',  # haa'
    'w',  # waaw
    'y',  # yaa'
    'v',  # /v/ for loanwords e.g. in u'fydyw': u'v i0 d y uu1',
    # vowels
    'a',  # short
    'u',
    'i',
    'aa',  # long
    'uu',
    'ii',
]



'''

EOS_TOKEN = '_eos_'
DOUBLING_TOKEN = '_dbl_'
SEPARATOR_TOKEN = '_+_'
EOS_TOKENS = [SEPARATOR_TOKEN, EOS_TOKEN]


symbols = [
    # special tokens
    EOS_TOKEN,  # eos-token
    '_sil_',  # silence
    DOUBLING_TOKEN,  # doubling
    SEPARATOR_TOKEN,  # word separator
    # consonants
    '<',  # hamza
    'b',  # baa'
    't',  # taa'
    '^',  # thaa'
    'j',  # jiim
    'H',  # Haa'
    'x',  # xaa'
    'd',  # daal
    '*',  # dhaal
    'r',  # raa'
    'z',  # zaay
    's',  # siin
    '$',  # shiin
    'S',  # Saad
    'D',  # Daad
    'T',  # Taa'
    'Z',  # Zhaa'
    'E',  # 3ayn
    'g',  # ghain
    'f',  # faa'
    'q',  # qaaf
    'k',  # kaaf
    'l',  # laam
    'm',  # miim
    'n',  # nuun
    'h',  # haa'
    'w',  # waaw
    'y',  # yaa'
    'v',  # /v/ for loanwords e.g. in u'fydyw': u'v i0 d y uu1',
    # vowels
    'a',  # short
    'u',
    'i',
    'aa',  # long
    'uu',
    'ii',
]

# Export all symbols:
symbols += [_pad] + list(_punctuation) + list(_letters)

# Special symbol ids
SPACE_ID = symbols.index(" ")
