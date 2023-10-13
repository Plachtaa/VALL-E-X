from vallex.utils.generation import SAMPLE_RATE, generate_audio, preload_models
from vallex.utils.prompt_making import make_prompt
from vallex.utils.download import download_models_from_github
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models()

text_prompt = """
Hey, Traveler, Listen to this, This machine has taken my voice, and now it can talk just like me!
"""

### Use given transcript
make_prompt(name="test", audio_prompt_path="test.wav",
            transcript="Enter Prompt Here")

audio_array = generate_audio(text_prompt, prompt="test")

write_wav("test.wav", SAMPLE_RATE, audio_array)
