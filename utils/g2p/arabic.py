from gruut import sentences
import re
import string
#===================== clean arabic text =========================#
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida

                         """, re.VERBOSE)

def remove_numbers(text):
    number_pattern = r'\d+'
    text = re.sub(pattern=number_pattern, repl=" ", string=text)
    return text

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text

def remove_extra_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\.\s+", ".", text)
    return text



def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def arabic_cleaner(text):
    text=normalize_arabic(text)
    text=remove_diacritics(text)
    text=remove_punctuations(text)
    text=remove_repeating_char(text)
    text=remove_numbers(text)
    text=remove_extra_space(text)
    return text

#=================== convert arabic text into ipa ===================#
def arabic_to_ipa(text):
    text=arabic_cleaner(text)
    phonemes = []
    for sent in sentences(text, lang="ar"):
        for word in sent:
            if hasattr(word, "phonemes") and word.phonemes:
                phonemes.extend(map(str, word.phonemes))

    phonemes_string = ''.join(phonemes)
    return phonemes_string





""" if __name__ == '__main__':
    text ="ظاهر"
    print(arabic_to_ipa(text))
  """
my_list = []  # Choose a different name
""" if __name__ == '__main__':
    text ="ظاهر"
    print(arabic_to_ipa(text))
 """

