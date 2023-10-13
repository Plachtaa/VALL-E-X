import nltk
import jieba
import sudachipy
import langid
langid.set_languages(['en', 'zh', 'ja'])

def split_text_into_sentences(text):
    if langid.classify(text)[0] == "en":
        sentences = nltk.tokenize.sent_tokenize(text)

        return sentences
    elif langid.classify(text)[0] == "zh":
        sentences = []
        segs = jieba.cut(text, cut_all=False)
        segs = list(segs)
        start = 0
        for i, seg in enumerate(segs):
            if seg in ["。", "！", "？", "……"]:
                sentences.append("".join(segs[start:i + 1]))
                start = i + 1
        if start < len(segs):
            sentences.append("".join(segs[start:]))

        return sentences
    elif langid.classify(text)[0] == "ja":
        sentences = []
        tokenizer = sudachipy.Dictionary().create()
        tokens = tokenizer.tokenize(text)
        current_sentence = ""

        for token in tokens:
            current_sentence += token.surface()
            if token.part_of_speech()[0] == "補助記号" and token.part_of_speech()[1] == "句点":
                sentences.append(current_sentence)
                current_sentence = ""

        if current_sentence:
            sentences.append(current_sentence)

        return sentences

    raise RuntimeError("It is impossible to reach here.")

long_text = """
This is a very long paragraph, so most TTS model is unable to handle it. Hence, we have to split it into several sentences. With the help of NLTK, we can split it into sentences. However, the punctuation is not preserved, so we have to add it back. How are we going to do write this code? Let's see. 
"""

long_text = """
现在我们要来尝试一下中文分句。因为很不幸的是，NLTK不支持中文分句。幸运的是，我们可以使用jieba来分句。但是，jieba分句后，标点符号会丢失，所以我们要手动添加回去。我现在正在想办法把这个例句写的更长更复杂一点，来测试jieba分句的性能。嗯......省略号，感觉不太好，因为省略号不是句号，所以jieba不会把它当作句子的结尾。会这样吗？我们来试试看。
"""

long_text = """
これなら、英語と中国語の分句もできる。でも、日本語はどうする？まつわ、ChatGPTに僕と教えてください。ちょーと待ってください。あ、出来た！
"""