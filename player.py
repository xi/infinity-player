"Play an infinite remix of your favorite songs."

from random import random
import argparse
import gzip
import os
import pickle
import struct

from scipy.misc import imresize
import alsaaudio
import librosa
import numpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'timbre.pickle'), 'rb') as fh:
    TIMBRE_PATTERNS = pickle.load(fh)


def enhance_diagonals(R, weight=0.2, steps=1):
    for i in range(steps):
        # combine each cell with its diagonal neighbors
        R1 = numpy.roll(R, (1, 1), (0, 1))
        R2 = numpy.roll(R, (-1, -1), (0, 1))
        R = (weight * (R1 + R2) + (1 - weight) * R) / 2
    return R


def iter_beat_slices(y, beat_frames):
    beat_samples = librosa.frames_to_samples(beat_frames)
    yield 0, beat_samples[0]
    for start, end in zip(beat_samples[0:-1], beat_samples[1:]):
        yield start, end
    yield beat_samples[-1], len(y) - 1


def timbre(y):
    spectrum = numpy.abs(librosa.stft(y))
    resized = imresize(spectrum, (50, 70))
    l = []
    for pattern in TIMBRE_PATTERNS:
        l.append(numpy.sum(pattern * resized))
    return l


def analyze(y, sample_rate, beat_frames, bins_per_octave=12, n_octaves=7):
    cqt = librosa.cqt(y=y, sr=sample_rate)
    C = librosa.amplitude_to_db(cqt, ref=numpy.max)
    sync = librosa.util.sync(C, beat_frames)
    R_cqt = librosa.segment.recurrence_matrix(sync, width=4, mode='affinity')

    tim = numpy.array([
        timbre(y[s:e]) for s, e in iter_beat_slices(y, beat_frames)
    ]).T
    R_timbre = librosa.segment.recurrence_matrix(tim, width=4, mode='affinity')

    return (R_cqt + R_timbre) / 2


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
    n = len(R)

    R = enhance_diagonals(R, 0.8, 4)

    # scale
    x_max = R.max()
    x_min = x_max * threshold
    y_max = (x_max + 0.5) / 2
    # print('mapping {},{} to {},{}'.format(x_min, x_max, 0, y_max))
    R_norm = (R - x_min) / (x_max - x_min) * y_max

    # privilege jumps back in order to prolong playing
    R *= numpy.ones((n, n)) * 0.9 + numpy.tri(n, k=-1) * 0.1

    # privilege wide jumps
    M = numpy.zeros((n, n))
    for i in range(1, n):
        M += numpy.tri(n, k=-i)
        M += numpy.tri(n, k=-i).T
    R *= (M / (n - 1)) ** 0.1

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


def plot(R):
    import librosa.display
    import matplotlib.pyplot as plt

    plt.figure()
    librosa.display.specshow(R)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.5, help='Between 0 and 1. '
        'A higher value will result in fewer but better jumps. (Default: 0.5)')
    parser.add_argument(
        '-f', '--force', action='store_true',
        help='Ignore previously saved analysis data.')
    parser.add_argument('-P', '--plot', action='store_true')
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

    if args.plot:
        plot(R)

    print('Playing… (Press Ctrl-C to stop)')
    play(buffers, sample_rate, jumps)


if __name__ == '__main__':
    main()
