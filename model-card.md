# Model Card: VALL-E X

**Author**: [Songting](https://github.com/Plachtaa).<br>
<br>
This is the official codebase for running open-sourced VALL-E X.

The following is additional information about the models released here.

## Model Details

VALL-E X is a series of two transformer models that turn text into audio.

### Phoneme to acoustic tokens
 - Input: IPAs converted from input text by a rule-based G2P tool.
 - Output: tokens from the first codebook of the [EnCodec Codec](https://github.com/facebookresearch/encodec) from facebook

### Coarse to fine tokens
 - Input: IPAs converted from input text by a rule-based G2P tool & the first codebook from EnCodec
 - Output: 8 codebooks from EnCodec

### Architecture
|          Model           | Parameters | Attention  | Output Vocab size |  
|:------------------------:|:----------:|------------|:-----------------:|
|         G2P tool         |     -      | -          |        69         |
| Phoneme to coarse tokens |   150 M    | Causal     |     1x 1,024      |
|  Coarse to fine tokens   |   150 M    | Non-causal |     7x 1,024      |

### Release date
August 2023

## Broader Implications
We anticipate that this model's text to audio capabilities can be used to improve accessbility tools in a variety of languages. 
Straightforward improvements will allow models to run faster than realtime, rendering them useful for applications such as virtual assistants. 