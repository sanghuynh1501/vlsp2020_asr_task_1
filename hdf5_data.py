import os
import h5py
import numpy as np
import random


class HDF5DatasetWriter:
    def __init__(self, audio_dims, label_dims, output_path, batch_size=8, buffer_size=10):
        if os.path.exists(output_path):
            raise ValueError(output_path)
        self.db = h5py.File(output_path, "w")
        self.audios = self.db.create_dataset("audios", audio_dims, dtype="float", compression="gzip",
                                             chunks=(batch_size, audio_dims[1], audio_dims[2]))

        self.labels = self.db.create_dataset("labels", label_dims, dtype="float", compression="gzip",
                                             chunks=(batch_size, label_dims[1]))

        self.audio_lens = self.db.create_dataset("audio_lens", (audio_dims[0], 1), dtype="int", compression="gzip",
                                                 chunks=(batch_size, 1))

        self.label_lens = self.db.create_dataset("label_lens", (audio_dims[0], 1), dtype="int", compression="gzip",
                                                 chunks=(batch_size, 1))

        self.bufSize = buffer_size
        self.buffer = {"audios": [], "labels": [], "audio_lens": [], "label_lens": []}
        self.idx = 0

    def add(self, audios, labels, audio_lens, label_lens):
        self.buffer["audios"].extend(audios)
        self.buffer["labels"].extend(labels)
        self.buffer["audio_lens"].extend(audio_lens)
        self.buffer["label_lens"].extend(label_lens)
        if len(self.buffer["audios"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["audios"])
        self.audios[self.idx:i] = self.buffer["audios"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.audio_lens[self.idx:i] = self.buffer["audio_lens"]
        self.label_lens[self.idx:i] = self.buffer["label_lens"]
        self.idx = i
        self.buffer = {"audios": [], "labels": [], "audio_lens": [], "label_lens": []}

    def close(self):
        if len(self.buffer["audios"]) > 0:
            self.flush()
        self.db.close()


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size, number=None):
        self.batchSize = batch_size
        self.db = h5py.File(db_path)
        self.numAudios = self.db["labels"].shape[0]
        self.indexes = []
        for i in range(self.numAudios):
            self.indexes.append(i)
        if number is not None:
            self.indexes = self.indexes[: number]

    def get_total_samples(self):
        return len(self.indexes)

    def generator(self):
        random.shuffle(self.indexes)
        for i in range(0,  len(self.indexes), self.batchSize):
            audios = self.db["audios"][self.indexes[i]: self.indexes[i] + self.batchSize]
            labels = self.db["labels"][self.indexes[i]: self.indexes[i] + self.batchSize]
            audio_lens = self.db["audio_lens"][self.indexes[i]: self.indexes[i] + self.batchSize]
            label_lens = self.db["label_lens"][self.indexes[i]: self.indexes[i] + self.batchSize]
            yield np.array(audios), np.array(labels), np.array(audio_lens), np.array(label_lens)

    def close(self):
        self.db.close()