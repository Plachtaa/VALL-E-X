"""
Description taken from official website: https://arabicspeech.org/mgb2/
The Multi-Dialect Broadcast News Arabic Speech Recognition (MGB-2):
The second edition of the Multi-Genre Broadcast (MGB-2) Challenge is
an evaluation of speech recognition and lightly supervised alignment
using TV recordings in Arabic. The speech data is broad and multi-genre,
spanning the whole range of TV output, and represents a challenging task for
speech technology. In 2016, the challenge featured two new Arabic tracks based
on TV data from Aljazeera. It was an official challenge at the 2016 IEEE
Workshop on Spoken Language Technology. The 1,200 hours MGB-2: from Aljazeera
TV programs have been manually captioned with no timing information.
QCRI Arabic ASR system has been used to recognize all programs. The ASR output
was used to align the manual captioning and produce speech segments for
training speech recognition. More than 20 hours from 2015 programs have been
transcribed verbatim and manually segmented. This data is split into a
development set of 10 hours, and a similar evaluation set of 10 hours.
Both the development and evaluation data have been released in the 2016 MGB
challenge
"""
from itertools import chain
from logging import info
from os import path, system
from pathlib import Path
from re import match, sub
from shutil import copy
from string import punctuation
from typing import Dict, Union

from lhotse import (
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    fix_manifests,
    validate_recordings_and_supervisions,
)
from lhotse.kaldi import load_kaldi_data_dir
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.utils import Pathlike, check_and_rglob, is_module_available, recursion_limit


def download_mgb2(
    target_dir: Pathlike = ".",
) -> None:
    """
    Download and untar the dataset.

    NOTE: This function just returns with a message since MGB2 is not available
    for direct download.

    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    """
    info(
        "MGB2 is not available for direct download. Please fill out the form"
        "at https://arabicspeech.org/mgb2 to download the corpus."
    )

    target_dir = Path(target_dir)
    corpus_dir = target_dir / "MGB2"
    target_dir.mkdir(parents=True, exist_ok=True)
     
    return corpus_dir


def prepare_mgb2(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    text_cleaning: bool = True,
    buck_walter: bool = False,
    num_jobs: int = 1,
    mer_thresh: int = 80,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param text_cleaning: Bool, if True, basic text cleaning is performed (similar to ESPNet recipe).
    :param buck_walter: Bool, use BuckWalter transliteration
    :param num_jobs: int, the number of jobs to use for parallel processing.
    :param mer_thresh: int, filter out segments based on mer (Match Error Rate)
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.

    .. note::
        Unlike other recipes, output_dir is not Optional here because we write the manifests
        to the output directory while processing to avoid OOM issues, since it is a large dataset.

    .. caution::
        The `text_cleaning` option removes all punctuation and diacritics.
    """

    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    dataset_parts = ["dev", "train", "test"]
    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Maybe the manifests already exist: we can read them and save a bit of preparation time.
        manifests = read_manifests_if_cached(
            dataset_parts=dataset_parts,
            output_dir=output_dir,
            prefix="mgb2",
            suffix="jsonl.gz",
            lazy=True,
        )

    for part in dataset_parts:
        info(f"Processing MGB2 subset: {part}")
        if manifests_exist(
            part=part, output_dir=output_dir, prefix="mgb2", suffix="jsonl.gz"
        ):
            info(f"MGB2 subset: {part} already prepared - skipping.")
            continue

        # Read the recordings and write them into manifest. We additionally store the
        # duration of the recordings in a dict which will be used later to create the
        # supervisions.

        output_dir = Path(output_dir)
        corpus_dir = Path(corpus_dir)
        if part == "test" or part == "dev":
            (output_dir / part).mkdir(parents=True, exist_ok=True)
            copy(
                corpus_dir / part / "text.non_overlap_speech",
                output_dir / part / "text",
            )
            copy(
                corpus_dir / part / "segments.non_overlap_speech",
                output_dir / part / "segments",
            )
            with open(corpus_dir / part / "wav.scp", "r") as f_in, open(
                output_dir / part / "wav.scp", "w"
            ) as f_out:
                for line in f_in:
                    f_out.write(line.replace("wav/", f"{corpus_dir}/{part}/wav/"))

            recordings, supervisions, _ = load_kaldi_data_dir(
                (output_dir / part), 16000
            )
            if buck_walter is False:
                supervisions = supervisions.transform_text(from_buck_walter)
            if part == "test":
                assert (
                    len(supervisions) == 5365
                ), f"Expected 5365 supervisions for test, found {len(supervisions)}"
            elif part == "dev":
                assert (
                    len(supervisions) == 5002
                ), f"Expected 5002 supervisions for dev, found {len(supervisions)}"
        elif part == "train":
            recordings = RecordingSet.from_dir(
                (corpus_dir / part / "wav"), pattern="*.wav", num_jobs=num_jobs
            )

            xml_paths = check_and_rglob(
                path.join(corpus_dir, part, "xml/utf8"), "*.xml"
            )
            # Read supervisions and write them to manifest
            with recursion_limit(5000):
                supervisions_list = list(
                    chain.from_iterable(
                        [make_supervisions(p, mer_thresh) for p in xml_paths]
                    )
                )

            supervisions = SupervisionSet.from_segments(supervisions_list)

            assert (
                len(supervisions) == 375103
            ), f"Expected 375103 supervisions for train, found {len(supervisions)}"

            if text_cleaning is True:
                supervisions = supervisions.transform_text(cleaning)

        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        # saving recordings and supervisions
        recordings.to_file((output_dir / f"mgb2_recordings_{part}.jsonl.gz"))
        supervisions.to_file((output_dir / f"mgb2_supervisions_{part}.jsonl.gz"))

        manifests[part] = {
            "recordings": recordings,
            "supervisions": supervisions,
        }
    return manifests


_unicode = (
    "\u0622\u0624\u0626\u0628\u062a\u062c\u06af\u062e\u0630\u0632"
    "\u0634\u0636\u0638\u063a\u0640\u0642\u0644\u0646\u0648\u064a\u064c\u064e"
    "\u0650\u0652\u0670\u067e\u0686\u0621\u0623\u0625\u06a4\u0627\u0629\u062b"
    "\u062d\u062f\u0631\u0633\u0635\u0637\u0639\u0641\u0643\u0645\u0647\u0649"
    "\u064b\u064d\u064f\u0651\u0671"
)
_buckwalter = "|&}btjGx*z$DZg_qlnwyNaio`PJ'><VApvHdrsSTEfkmhYFKu~{"

_backward_map = {ord(b): a for a, b in zip(_unicode, _buckwalter)}


def from_buck_walter(s: str) -> str:
    return s.translate(_backward_map)


def remove_diacritics(text: str) -> str:
    # https://unicode-table.com/en/blocks/arabic/
    return sub(r"[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+", "", text)


def remove_punctuations(text: str) -> str:
    """This function  removes all punctuations except the verbatim"""

    arabic_punctuations = """﴿﴾`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
    english_punctuations = punctuation
    # remove all non verbatim punctuations
    all_punctuations = set(arabic_punctuations + english_punctuations)

    for p in all_punctuations:
        if p in text:
            text = text.replace(p, " ")
    return text


def remove_non_alphanumeric(text: str) -> str:
    text = text.lower()
    return sub(r"[^\u0600-\u06FF\s\da-z]+", "", text)


def remove_single_char_word(text: str) -> str:
    """
    Remove single character word from text
    Example: I am in a a home for two years => am in home for two years
    Args:
            text (str): text
    Returns:
            (str): text with single char removed
    """
    words = text.split()

    filter_words = [word for word in words if len(word) > 1 or word.isnumeric()]
    return " ".join(filter_words)


def east_to_west_num(text: str) -> str:
    eastern_to_western = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "٪": "%",
        "_": " ",
        "ڤ": "ف",
        "|": " ",
    }
    trans_string = str.maketrans(eastern_to_western)
    return text.translate(trans_string)


def remove_extra_space(text: str) -> str:
    text = sub(r"\s+", " ", text)
    text = sub(r"\s+\.\s+", ".", text)
    return text


def cleaning(text: str) -> str:
    text = remove_punctuations(text)
    text = east_to_west_num(text)
    text = remove_diacritics(text)
    text = remove_non_alphanumeric(text)
    text = remove_single_char_word(text)
    text = remove_extra_space(text)
    return text


def make_supervisions(xml_path: str, mer_thresh: int) -> None:
    if not is_module_available("bs4"):
        raise ValueError(
            "To prepare MGB2 data, please 'pip install beautifulsoup4' first."
        )
    from bs4 import BeautifulSoup

    xml_handle = open(xml_path, "r")
    soup = BeautifulSoup(xml_handle, "xml")
    return [
        SupervisionSegment(
            id=segment["id"] + "_" + segment["starttime"] + ":" + segment["endtime"],
            recording_id=segment["id"].split("_utt")[0].replace("_", "-"),
            start=float(segment["starttime"]),
            duration=round(
                float(segment["endtime"]) - float(segment["starttime"]), ndigits=8
            ),
            channel=0,
            text=" ".join(
                [
                    element.string
                    for element in segment.find_all("element")
                    if element.string is not None
                ]
            ),
            language="Arabic",
            speaker=int(match(r"\w+speaker(\d+)\w+", segment["who"]).group(1)),
        )
        for segment in soup.find_all("segment")
        if mer_thresh is None or float(segment["WMER"]) <= mer_thresh
    ]
