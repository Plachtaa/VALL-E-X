from gruut import sentences

def arabic_to_api(text):
    phonemes = []
    for sent in sentences(text, lang="ar"):
        for word in sent:
            if hasattr(word, "phonemes") and word.phonemes:
                phonemes.extend(map(str, word.phonemes))

    phonemes_string = ''.join(phonemes)
    return phonemes_string

""" if __name__ == '__main__':
    text = "ذهب علي الى المدرسة"
    print(arabic_to_api(text))
 """