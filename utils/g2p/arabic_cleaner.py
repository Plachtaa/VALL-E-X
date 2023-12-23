import re
import os
import pickle
"""
def _normalize_data(text):
    rel_path = os.path.dirname(__file__)
    norm_dict_path = os.path.join(rel_path, "dictionaries/norm_dict.pl")
    norm_dict = pickle.load(open(norm_dict_path, "rb"))
    # use a mapping dictionary
    regex = re.compile("|".join(map(re.escape, norm_dict.keys())))
    text = regex.sub(lambda match: norm_dict[match.group(0)], text)
    return text
"""

def _remove_english_chars(text):
    return re.sub("[a-zA-Z]", "", text)

def _remove_digits(text):
    return re.sub("[0-9]", "", text)

def _remove_all_english(text):
    return re.sub("[a-zA-Z0-9]", "", text)


def _keep_only_arabic_chars(text):
    text = re.compile("([\n\u0621-\u064A0-9])").sub(r" ", text)
    return text

def _remove_extra_spaces(text):
    text = re.sub(" +", " ", text)
    return text


def _remove_repeated_chars(text):
    return re.sub(r"(.)\1+", r"\1\1", text)


def clean_arabic_text(text):
    #text=_normalize_data(text)
    text=_remove_all_english(text)
    text=_keep_only_arabic_chars(text)
    text=_remove_digits(text)
    text=_remove_extra_spaces(text)
    text=_remove_repeated_chars(text)
    text=_remove_english_chars(text)
    return text
""" if __name__=="__main__":
    text = "السلام عليكم و رحمة الله و بركاته"
    print(text)
    clean_text=clean_arabic_text(text)
    print(clean_text)"""