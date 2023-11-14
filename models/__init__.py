import argparse

import torch.nn as nn
# from icefall.utils import AttributeDict, str2bool

from .macros import (
    NUM_AUDIO_TOKENS,
    NUM_MEL_BINS,
    NUM_SPEAKER_CLASSES,
    NUM_TEXT_TOKENS,
    SPEAKER_EMBEDDING_DIM,
)
from .transformer import Transformer
from .vallex import VALLE, VALLF
from .visualizer import visualize


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-name",
        type=str,
        default="VALL-E",
        help="VALL-E, VALL-F, Transformer.",
    )
    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=1024,
        help="Embedding dimension in the decoder model.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=16,
        help="Number of attention heads in the Decoder layers.",
    )
    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=12,
        help="Number of Decoder layers.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Model scale factor which will be assigned different meanings in different models.",
    )
    parser.add_argument(
        "--norm-first",
        type=bool,
        default=True,
        help="Pre or Post Normalization.",
    )
    parser.add_argument(
        "--add-prenet",
        type=bool,
        default=False,
        help="Whether add PreNet after Inputs.",
    )

    # VALL-E & F
    parser.add_argument(
        "--prefix-mode",
        type=int,
        default=1,
        help="The mode for how to prefix VALL-E NAR Decoder, "
        "0: no prefix, 1: 0 to random, 2: random to random, 4: chunk of pre or post utterance.",
    )
    parser.add_argument(
        "--share-embedding",
        type=bool,
        default=True,
        help="Share the parameters of the output projection layer with the parameters of the acoustic embedding.",
    )
    parser.add_argument(
        "--prepend-bos",
        type=bool,
        default=False,
        help="Whether prepend <BOS> to the acoustic tokens -> AR Decoder inputs.",
    )
    parser.add_argument(
        "--num-quantizers",
        type=int,
        default=8,
        help="Number of Audio/Semantic quantization layers.",
    )

    # Transformer
    parser.add_argument(
        "--scaling-xformers",
        type=bool,
        default=False,
        help="Apply Reworked Conformer scaling on Transformers.",
    )


def get_model(params) -> nn.Module:
    if params.model_name.lower() in ["vall-f", "vallf"]:
        model = VALLF(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            prefix_mode=params.prefix_mode,
            share_embedding=params.share_embedding,
            nar_scale_factor=params.scale_factor,
            prepend_bos=params.prepend_bos,
            num_quantizers=params.num_quantizers,
        )
    elif params.model_name.lower() in ["vall-e", "valle"]:
        model = VALLE(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            prefix_mode=params.prefix_mode,
            share_embedding=params.share_embedding,
            nar_scale_factor=params.scale_factor,
            prepend_bos=params.prepend_bos,
            num_quantizers=params.num_quantizers,
        )
    else:
        assert params.model_name in ["Transformer"]
        model = Transformer(
            params.decoder_dim,
            params.nhead,
            params.num_decoder_layers,
            norm_first=params.norm_first,
            add_prenet=params.add_prenet,
            scaling_xformers=params.scaling_xformers,
        )

    return model
