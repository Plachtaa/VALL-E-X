from vallex.utils.generation import SAMPLE_RATE, generate_audio, preload_models
from vallex.utils.prompt_making import make_prompt
from vallex.utils.download import download_models_from_github
from scipy.io.wavfile import write as write_wav
from playsound import playsound

# download and load all models
model, codec, vocos = preload_models()


while True:
    audio_array = generate_audio(model, codec, vocos, input("what to say: "))
    write_wav("test.wav", SAMPLE_RATE, audio_array)
    try:
        playsound('test.wav')
    except:
        print("Failed to play")

