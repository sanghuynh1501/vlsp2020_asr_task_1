import os
import numpy as np
from tqdm import tqdm
import librosa
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from hdf5_data import HDF5DatasetWriter

f = open('labels/tlabels.txt', "r")

pad_token = '<PAD>'
start_token = '<SOS>'
end_token = '<EOS>'
mfcc_feature = 39

MAX_LEN_TEXT = 200
MAX_LEN_AUDIO_ORIGIN = 200000
MAX_LEN_AUDIO = 391
FOLDER_NAME = "data"
LABELS = f.read().split(',')

print(LABELS)

data_list = os.listdir('data')
tokens = []

audio_links = []
label_links = []


def pad_audio(samples, length=MAX_LEN_AUDIO):
    if len(samples) >= length:
        return samples
    else:
        while len(samples) < length:
            samples = np.concatenate([samples, np.zeros((1, mfcc_feature))], 0)
        return samples


def text_to_labels(text):
    one_hot = []
    texts = text.split('_')
    for char in texts:
        if char not in LABELS:
            for c in char:
                one_hot.append(LABELS.index(c))
        else:
            one_hot.append(LABELS.index(char))

    end_token = one_hot[-1]
    one_hot = one_hot[:-1]
    while len(one_hot) < MAX_LEN_TEXT + 1:
        one_hot.append(LABELS.index(pad_token))
    one_hot.append(end_token)

    return np.array(one_hot)


def filter_data(file_name):
    if '.wav' in file_name:
        _, data = wavfile.read(file_name)
        txt_file = file_name.split('.')[0] + '.txt'
        f = open(txt_file, "r")

        text = f.read()
        text = text.strip()
        text = text.replace('\n', '')
        text = text.replace('\'', '')
        text = start_token + '_' + text
        text = text + '_' + end_token

        if len(text) <= MAX_LEN_TEXT + len(start_token) + len(end_token) + 2 and MAX_LEN_AUDIO_ORIGIN >= len(data) > 0:
            audio_links.append(file_name)
            label_links.append(txt_file)


def read_data(wav_file, txt_file):
    sample_rate, data = wavfile.read(wav_file)

    f = open(txt_file, "r")
    text = f.read()
    text = text.strip()
    text = text.replace('\n', '')
    text = text.replace('\'', '')
    text_length = len(text) + 1
    text = start_token + '_' + text
    text = text + '_' + end_token

    data = np.reshape(data, (data.shape[0],))
    data = data.astype(np.float32)
    data = data / (2.0 ** (16 - 1) + 1)

    S = librosa.feature.melspectrogram(data, sr=sample_rate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=mfcc_feature)
    mfcc = np.reshape(mfcc, (mfcc.shape[1], mfcc.shape[0]))
    length = len(mfcc)

    mfcc = pad_audio(mfcc)
    # mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)

    label = text_to_labels(text)
    label = np.expand_dims(label, 0)

    return mfcc, label, np.array([[length]]), np.array([[text_length]])


def dum_data(audioes, labels, path):
    length = len(audioes)
    dump_data = HDF5DatasetWriter((length, MAX_LEN_AUDIO, mfcc_feature), (length, MAX_LEN_TEXT + 2), path)

    with tqdm(total=length) as pbar:
        for _audio, _label in zip(audioes, labels):
            c, _label, _len, _text_len = read_data(_audio, _label)
            dump_data.add(c, _label, _len, _text_len)
            pbar.update(1)


for container in data_list:
    folder_name = FOLDER_NAME + '/' + container
    for file_name in os.listdir(folder_name):
        filter_data(folder_name + '/' + file_name)

audio_train, audio_test, label_train, label_test = train_test_split(audio_links, label_links, test_size=0.1,
                                                                    random_state=42)

print(len(audio_train))
print(len(audio_test))
print(len(label_train))
print(len(label_test))

dum_data(audio_train[:135140], label_train[:135140], 'train.hdf5')
dum_data(audio_test[:15010], label_test[:15010], 'test.hdf5')
