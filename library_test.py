from vallex.utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
model, codec, vocos = preload_models()


while True:
    audio_array = generate_audio(model, codec, vocos, input("what to say: "))
    write_wav("test1.wav", SAMPLE_RATE, audio_array)

