import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa

from hdf5_data import HDF5DatasetGenerator

devices = tf.config.experimental.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(devices[0], True)
except IndexError:
    print('error')

from model import Transformer


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


f = open('labels/tlabels.txt', "r")
LABELS = f.read().split(',')

END_TOKEN = '<end>'
START_TOKEN = '<start>'

d_model = 256
batch_size = 8

train_data = HDF5DatasetGenerator('train.hdf5', batch_size, 1000)
test_data = HDF5DatasetGenerator('test.hdf5', batch_size, 1000)

train_total = train_data.get_total_samples()
test_total = test_data.get_total_samples()

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction='none')


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def lableSmoothingLoss(real, pred, vocab_size, epsilon):
    """
    pred (FloatTensor): batch_size x vocab_size
    real (LongTensor): batch_size
    """
    real = tf.cast(real, tf.int32)
    # print(real)
    real_smoothed = label_smoothing(tf.one_hot(real, depth=vocab_size), epsilon)
    # print(real_smoothed)
    # mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real_smoothed, pred)

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)  # 转换为与loss相同的类型

    # Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
    loss_ *= mask

    return tf.reduce_mean(loss_)


def sum_predict(ctc, decode):
    padding = np.zeros((decode.shape[0], decode.shape[1], 1))
    decode = np.concatenate([decode, padding], 2)
    padding = np.zeros((decode.shape[0], ctc.shape[1] - decode.shape[1], decode.shape[2]))
    for i in range(len(padding)):
        padding[i, :, 2] = 1
    decode = np.concatenate([decode, padding], 1)

    return decode * 0.8 + (1 - 0.8) * ctc


def decode_index(labels):
    text = ''
    labels = labels.astype(np.int64)
    for index in labels:
        if index < len(LABELS):
            text += LABELS[index]
    return text


def decode_index_argmax(labels):
    return decode_index(np.argmax(labels, -1))


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def ctc_loss_function(labels, logits, logit_length, label_length):
    loss = tf.keras.backend.ctc_batch_cost(labels, logits, logit_length, label_length)
    return tf.reduce_sum(loss)


transformer = Transformer(
    input_vocab_size=320,
    num_layers=4, d_model=d_model, num_heads=4, dff=1024,
    target_vocab_size=len(LABELS),
    pe_input=10000, pe_target=6000
)


def create_audio_padding_mask(seqs, length):
    masks = []
    for idx, seq in enumerate(seqs):
        len_audio = length[idx][0]
        mask_zeros = np.zeros((len_audio,))
        mask_ones = np.ones((len(seq) - len_audio,))
        mask = np.concatenate([mask_zeros, mask_ones], 0)
        masks.append(mask)
    masks = np.array(masks)
    # add extra dimensions to add the padding
    # to the attention logits.
    return masks[:, tf.newaxis, tf.newaxis, :]


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar, length):
    # Encoder padding mask
    enc_padding_mask = create_audio_padding_mask(inp, length)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_audio_padding_mask(inp, length)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "speech_transformer_ctc_checkpoint"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


def train_step(audios, tar, length, text_length):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(audios, tar_inp, length)

    with tf.GradientTape() as tape:
        ctc_output, predictions, _ = transformer(audios, tar_inp,
                                                 True,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
        pred_loss = lableSmoothingLoss(tar_real, predictions, len(LABELS), 0.1)
        ctc_loss = ctc_loss_function(tar_real, ctc_output, np.ones((audios.shape[0], 1)) * audios.shape[1], text_length)

        loss = 0.999 * pred_loss + (1 - 0.999) * ctc_loss

    predictions = sum_predict(ctc_output, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    padding = np.ones((audios.shape[0], predictions.shape[1] - tar_real.shape[1])) * 2
    tar_real = np.concatenate([tar_real, padding], 1)
    train_accuracy(tar_real, predictions)


def test_audios_sample(audios, tar, length):
    tar_inp = tar[:, :-1]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(audios, tar_inp, length)

    ctc_output, predictions, _ = transformer(audios, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)

    tar_text = decode_index(tar_inp[0])
    predictions = sum_predict(ctc_output, predictions)
    pre_text = decode_index_argmax(predictions.numpy()[0])
    print('tar_text ', tar_text)
    print('pre_text ', pre_text)


def test_step(audios, tar, length, text_length):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(audios, tar_inp, length)

    ctc_output, predictions, _ = transformer(audios, tar_inp,
                                             False,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)

    pred_loss = lableSmoothingLoss(tar_real, predictions, len(LABELS), 0.1)
    ctc_loss = ctc_loss_function(tar_real, ctc_output, np.ones((audios.shape[0], 1)) * audios.shape[1], text_length)
    predictions = sum_predict(ctc_output, predictions)
    loss = 0.999 * pred_loss + (1 - 0.999) * ctc_loss

    test_loss(loss)
    padding = np.ones((audios.shape[0], predictions.shape[1] - tar_real.shape[1])) * 2
    tar_real = np.concatenate([tar_real, padding], 1)
    test_accuracy(tar_real, predictions)


def padding_data(audios, labels, audio_len, text_len):
    max_audio_len = np.max(audio_len)
    max_text_len = np.max(text_len)

    audios_batch = audios[:, :max_audio_len, :]
    labels_batch = np.concatenate([labels[:, :max_text_len], np.ones((len(labels), 1)) * 2], 1)

    while audios_batch.shape[1] < max_text_len:
        padding = np.zeros((audios.shape[0], 1, audios.shape[2]))
        audios_batch = np.concatenate([audios_batch, padding], 1)

    return audios_batch, labels_batch


min_loss = float('inf')

for epoch in range(1000):
    start = time.time()

    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

    test_audios = np.array([])
    test_labels = np.array([])
    test_length = np.array([])

    with tqdm(total=train_total) as pbar:
        for audios, labels, length, text_length in train_data.generator():
            audios, labels = padding_data(audios, labels, length, text_length)
            train_step(audios, labels, length, text_length)
            pbar.update(batch_size)
            test_audios = audios
            test_labels = labels
            test_length = length

    with tqdm(total=test_total) as pbar:
        for audios, labels, length, text_length in test_data.generator():
            audios, labels = padding_data(audios, labels, length, text_length)
            test_step(audios, labels, length, text_length)
            pbar.update(batch_size)

    test_audios_sample(test_audios[: 2], test_labels[: 2], test_length[: 2])

    print('Epoch {} Train Loss {:.4f} Test Loss {:.4f} Train Acc {:.4f} Test Acc {:.4f}'.format(
        epoch + 1, train_loss.result(), test_loss.result(), train_accuracy.result(), test_accuracy.result()))

    if test_loss.result() < min_loss:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))
        min_loss = test_loss.result()
