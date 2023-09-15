"""
This code demonstrates how to run VALL-E-X with the pipeline-ai SDK.
This allows you to run VALL-E-X on the cloud with a single command, and
a demo is avaible here: https://www.mystic.ai/paulh/vall-e-x/play.
"""

import pathlib
from pipeline import Pipeline, Variable, pipe
from pipeline.objects.graph import Directory
from pipeline.cloud import compute_requirements, environments, pipelines
from pipeline.objects import File
from pipeline.configuration import current_configuration

import utils.generation
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

current_configuration.set_debug_mode(True)

uploaded_dir = False


@pipe(on_startup=True, run_once=True)
def load(model_weights: Directory):
    utils.generation.checkpoints_dir = str(model_weights.path)
    preload_models()


@pipe
def predict(text: str) -> File:
    audio_array = generate_audio(text)

    write_wav("output.wav", SAMPLE_RATE, audio_array)
    return File(path="output.wav")


with Pipeline() as builder:
    text = Variable(
        str,
        title="Input text",
        description="The input text to create speech",
    )

    if uploaded_dir:
        # Make sure you have the checkpoint directory downloaded then run:
        # pipeline create files ./checkpoints -f
        # to upload the checkpoints directory to the cloud and get an ID

        model_dir = Directory.from_remote(
            id="file_96021134edca4ee1a181e80dd22631b7",
        )
    else:  # OR
        model_dir = Directory("./checkpoints")

    load(model_dir)

    output = predict(
        text,
    )

    builder.output(output)

my_pl = builder.get_pipeline()


requirements_path = pathlib.Path("requirements.txt")

env_id = environments.create_environment(
    "paulh/vall-e-x",
    python_requirements=requirements_path.read_text().splitlines(),
)


remote_pipeline = pipelines.upload_pipeline(
    my_pl,
    "paulh/vall-e-x",
    environment_id_or_name="paulh/vall-e-x",
    required_gpu_vram_mb=5_000,
    accelerators=[
        compute_requirements.Accelerator.nvidia_l4,
    ],
    modules=[
        "models",
        "modules",
        "data",
        "utils",
        "utils.g2p",
    ],
)

pipelines.run_pipeline(remote_pipeline.id, "Hello my name is paul")
