from random import random
import argparse
import gzip
import os
import pickle
import struct

import alsaaudio
import librosa
import numpy


def iter_beat_slices(y, beat_frames):
    beat_samples = librosa.frames_to_samples(beat_frames)
    yield 0, beat_samples[0]
    for start, end in zip(beat_samples[0:-1], beat_samples[1:]):
        yield start, end
    yield beat_samples[-1], len(y) - 1


def analyze(y, sample_rate, beat_frames, bins_per_octave=12, n_octaves=7):
    cqt = librosa.cqt(y=y, sr=sample_rate)
    C = librosa.amplitude_to_db(cqt, ref=numpy.max)
    sync = librosa.util.sync(C, beat_frames)
    return librosa.segment.recurrence_matrix(sync, width=4, mode='affinity')


def load(filename, force=False):
    y, sample_rate = librosa.load(filename, sr=None)

    fn_inf = filename + '.inf'
    if not force and os.path.exists(fn_inf):
        with gzip.open(fn_inf, 'rb') as fh:
            beat_frames, R = pickle.load(fh)
    else:
        print('Analyzing…')
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate)
        R = analyze(y, sample_rate, beat_frames)

        with gzip.open(fn_inf, 'wb') as fh:
            pickle.dump((beat_frames, R), fh)

    return y, sample_rate, beat_frames, R


def compute_buffers(y, beat_frames):
    int_max = numpy.iinfo(numpy.int16).max
    raw = (y * int_max).astype(numpy.int16).T.copy(order='C')

    buffers = []
    for start, end in iter_beat_slices(raw, beat_frames):
        samples = raw[start:end]
        data = struct.pack("h" * len(samples), *samples)
        duration = librosa.samples_to_time(end - start)
        buffers.append((data, duration))

    return buffers


def normalize(R, threshold):
    x_max = R.max()
    x_min = x_max * threshold
    y_max = (x_max + 0.5) / 2
    # print('mapping {},{} to {},{}'.format(x_min, x_max, 0, y_max))
    R_norm = (R - x_min) / (x_max - x_min) * y_max

    return R_norm * (R_norm > 0)


def compute_jumps(R):
    jumps = []
    for row in R:
        l = [(i, p) for i, p in enumerate(row) if p > 0]
        jumps.append(sorted(l, key=lambda x: -x[1]))
    return jumps


def play(buffers, sample_rate, jumps):
    # https://larsimmisch.github.io/pyalsaaudio/libalsaaudio.html#pcm-objects
    pcm = alsaaudio.PCM()
    pcm.setrate(sample_rate)
    pcm.setchannels(1)

    i = 0
    n = len(buffers)

    while True:
        data, duration = buffers[i]
        pcm.write(data)

        for j, p in jumps[i]:
            if p > random():
                # print('jump', i, j)
                i = j
                break

        i = i + 1

        if i >= n:
            # print('reached end')
            i = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-f', '--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    print('Loading', args.filename)
    y, sample_rate, beat_frames, R = load(args.filename, args.force)
    R = normalize(R, args.threshold)
    buffers = compute_buffers(y, beat_frames)
    jumps = compute_jumps(R)
    jump_count = sum(len(row) for row in jumps)

    print('Detected {} jump opportunities on {} beats'.format(
        jump_count, len(buffers)))
    print('Playing… (Press Ctrl-C to stop)')
    play(buffers, sample_rate, jumps)


if __name__ == '__main__':
    main()
