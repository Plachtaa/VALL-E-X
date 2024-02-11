from bs4 import BeautifulSoup
from pydub import AudioSegment
import os
from scipy.signal import butter, filtfilt
import numpy as np

def parse_xml(xml_file):
    segments = []
    with open(xml_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')
        for segment in soup.find_all('segment'):
            segment_data = {}
            segment_data['id'] = segment['id']
            segment_data['starttime'] = float(segment['starttime'])
            segment_data['endtime'] = float(segment['endtime'])
            segment_data['who'] = segment['who']
            words = [element.text for element in segment.find_all('element')]
            segment_data['words'] = ' '.join(words)
            segments.append(segment_data)
    return segments

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff_freq, fs):
    b, a = butter_lowpass(cutoff_freq, fs)
    y = filtfilt(b, a, data)
    return y

def remove_noise(audio_segment, cutoff_freq=4000):
    samples = np.array(audio_segment.get_array_of_samples())
    sample_rate = audio_segment.frame_rate
    filtered_samples = apply_lowpass_filter(samples, cutoff_freq, sample_rate)
    return AudioSegment(filtered_samples.tobytes(), frame_rate=sample_rate, sample_width=audio_segment.sample_width, channels=audio_segment.channels)

def split_audio_segments(input_file, start, end, dataset_part):
    sound = AudioSegment.from_wav(input_file)
    segment_duration = (end - start) * 1000  
    total_segments = len(sound) // segment_duration
    segments = []
    for i in range(total_segments + 1):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, len(sound))  
        segment = sound[start_time:end_time]
        segment = remove_noise(segment)  
        segments.append(segment)
        os.makedirs(os.path.join(f"mgb2_dataset/{dataset_part}", "wav"), exist_ok=True)
        segment.export(os.path.join(f"mgb2_dataset/{dataset_part}", "wav", f"wav_{i}.wav"), format="wav")

def create_mgb2_dataset(dataset_part, xml_utf8, wav_dir):
    # create txt folder
    try:
       os.makedirs(os.path.join("mgb2_dataset/", dataset_part, "txt"), exist_ok=True) 
       print("Text directory created successfully.")
       files = os.listdir(xml_utf8)
       for file in files:
         xml_file = os.path.join(xml_utf8, file)
         segments = parse_xml(xml_file)
         count = 0
         for segment in segments:
            words = segment['words']
            file_path = os.path.join("mgb2_dataset/", dataset_part, "txt", f"text_{count+1}.txt")  
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(words)
            print(f"File '{file_path}' created successfully.")
            count += 1
    except FileExistsError:
       print("Text directory already exists.")

      #  create wav folder
    try:
       os.makedirs(os.path.join("mgb2_dataset/", dataset_part, "wav"), exist_ok=True)  
       print("WAV directory created successfully.")
       wav_files = os.listdir(wav_dir)  
       for wav_file in wav_files:
         xml_file = os.path.join(xml_utf8, os.path.splitext(wav_file)[0] + ".xml")  
         segments = parse_xml(xml_file)
         for segment in segments:
            start = int(segment['starttime'])
            end = int(segment['endtime'])
            split_audio_segments(os.path.join(wav_dir, wav_file), start, end, dataset_part)
        
    except FileExistsError:
        print("WAV directory already exists.")
    
if __name__ == "__main__":
   dataset_part = "train"
   xml_utf8 = "D:\\MachineCourse\\Graduation_Project\\dev\\xml\\utf8"
   wav_dir = "D:\\MachineCourse\\Graduation_Project\\dev\\wav"
   create_mgb2_dataset(dataset_part, xml_utf8, wav_dir)
