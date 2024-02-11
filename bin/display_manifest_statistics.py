#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.             (authors: Fangjun Kuang)
# Copyright    2023                           (authors: Feiteng Li)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file displays duration statistics of utterances in the manifests.
You can use the displayed value to choose minimum/maximum duration
to remove short and long utterances during the training.
"""

import argparse
from pathlib import Path

from lhotse import load_manifest_lazy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized manifests.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    manifest_dir = args.manifest_dir or Path("data/tokenized")
    for part in ["train", "dev", "test"]:
        print(f"##  {part}")
        cuts = load_manifest_lazy(manifest_dir / f"cuts_{part}.jsonl.gz")
        cuts.describe()
        print("\n")


if __name__ == "__main__":
    main()
