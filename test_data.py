import os
import librosa
import numpy as np
from scipy.io import wavfile

f = open('labels/tlabels.txt', "r")

pad_token = '<PAD>'
start_token = '<SOS>'
end_token = '<EOS>'

LABELS = f.read().split(',')
MAX_LEN_TEXT = 200
MAX_LEN_AUDIO_ORIGIN = 200000
FOLDER_NAME = "data"
data_list = os.listdir('data')

total = 0
total_filter = 0

max_audio_len = 0
max_audio_link = ''

for container in data_list:
    folder_name = FOLDER_NAME + '/' + container
    for file_name in os.listdir(folder_name):
        if '.wav' in file_name:
            _, data = wavfile.read(folder_name + '/' + file_name)
            txt_file = folder_name + '/' + file_name.split('.')[0] + '.txt'
            f = open(txt_file, "r")

            text = f.read()
            text = text.strip()
            text = text.replace('\n', '')
            text = text.replace('\'', '')

            text = start_token + '_' + text
            text = text + '_' + end_token

            if len(text) <= MAX_LEN_TEXT + len(start_token) + len(end_token) + 2 and MAX_LEN_AUDIO_ORIGIN >= len(data) > 0:
                total_filter += 1
                if len(data) > max_audio_len:
                    max_audio_len = len(data)
                    max_audio_link = folder_name + '/' + file_name

            total += 1

print('max_audio_link ', max_audio_link)
print('max_audio_len ', max_audio_len)
print('percent ', (total_filter / total) * 100)

sample_rate, samples = wavfile.read(max_audio_link)
samples = samples / (2.0 ** (16 - 1) + 1)
S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)
mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=39)
mfcc = np.reshape(mfcc, (mfcc.shape[1], mfcc.shape[0]))

print('mfcc.shape ', mfcc.shape)


