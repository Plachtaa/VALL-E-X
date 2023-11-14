""" from https://github.com/keithito/tacotron """
import utils.g2p.cleaners
from utils.g2p.symbols import symbols
from tokenizers import Tokenizer

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


class PhonemeBpeTokenizer:
  def __init__(self, tokenizer_path = "./utils/g2p/bpe_1024.json"):
    self.tokenizer = Tokenizer.from_file(tokenizer_path)

  def tokenize(self, text):
    # 1. convert text to phoneme
    phonemes, langs = _clean_text(text, ['cje_cleaners'])
    # 2. replace blank space " " with "_"
    phonemes = phonemes.replace(" ", "_")
    # 3. tokenize phonemes
    phoneme_tokens = self.tokenizer.encode(phonemes).ids
    assert(len(phoneme_tokens) == len(langs))
    if not len(phoneme_tokens):
      raise ValueError("Empty text is given")
    return phoneme_tokens, langs

def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  symbol_to_id = {s: i for i, s in enumerate(symbols)}
  clean_text = _clean_text(text, cleaner_names)
  for symbol in clean_text:
    if symbol not in symbol_to_id.keys():
      continue
    symbol_id = symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = [_symbol_to_id[symbol] for symbol in cleaned_text if symbol in _symbol_to_id.keys()]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    s = _id_to_symbol[symbol_id]
    result += s
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(utils.g2p.cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text, langs = cleaner(text)
  return text, langs
