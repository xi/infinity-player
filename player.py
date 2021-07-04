"Play an infinite remix of your favorite songs."

from random import random
import argparse
import gzip
import os
import pickle
import shutil

from PIL import Image
import librosa
import numpy
import soundcard

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'timbre.pickle'), 'rb') as fh:
    TIMBRE_PATTERNS = pickle.load(fh)


def print_progress(i, n):
    cols, lines = shutil.get_terminal_size()
    pos = i * (cols - 5) // n
    s = ''
    for x in range(cols - 5):
        if x == pos:
            s += '|'
        elif x < pos:
            s += '='
        else:
            s += '-'
    s += f' {i:>4}'
    print(s, end='\r')


def enhance_diagonals(jumps, weight=0.2, steps=1):
    for i in range(steps):
        # combine each cell with its diagonal neighbors
        jumps1 = numpy.roll(jumps, (1, 1), (0, 1))
        jumps2 = numpy.roll(jumps, (-1, -1), (0, 1))
        jumps = (weight * (jumps1 + jumps2) + (1 - weight) * jumps) / 2
    return jumps


def iter_beat_slices(y, beat_frames):
    beat_samples = librosa.frames_to_samples(beat_frames)
    beat_samples = [0, *beat_samples, len(y) - 1]
    for start, end in zip(beat_samples[0:-1], beat_samples[1:]):
        yield start, end


def timbre(y):
    spectrum = numpy.abs(librosa.stft(y))
    resized = numpy.array(Image.fromarray(spectrum).resize((70, 50)))

    k = len(TIMBRE_PATTERNS)
    T = numpy.zeros((k, k))
    s = numpy.zeros((k, 1))

    for i, pattern in enumerate(TIMBRE_PATTERNS):
        s[i][0] = numpy.sum(TIMBRE_PATTERNS[i] * resized)
        for j, pattern2 in enumerate(TIMBRE_PATTERNS):
            T[i][j] = numpy.sum(pattern * pattern2)

    return numpy.linalg.inv(T) @ s


def analyze(y, sample_rate, beat_frames, bins_per_octave=12, n_octaves=7):
    # cqt = librosa.cqt(y=y, sr=sample_rate)
    # C = librosa.amplitude_to_db(cqt, ref=numpy.max)
    # sync = librosa.util.sync(C, beat_frames)
    # R_cqt = librosa.segment.recurrence_matrix(sync, width=4, mode='affinity')
    # return (R_cqt + R_timbre) / 2

    tim = numpy.array([
        timbre(y[start:end]) for start, end in iter_beat_slices(y, beat_frames)
    ]).T
    return librosa.segment.recurrence_matrix(tim, width=4, mode='affinity')


def load(filename, force=False):
    y, sample_rate = librosa.load(filename)

    fn_inf = filename + '.inf'
    if not force and os.path.exists(fn_inf):
        with gzip.open(fn_inf, 'rb') as fh:
            beat_frames, jumps = pickle.load(fh)
    else:
        print('Analyzing…')
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate)
        jumps = analyze(y, sample_rate, beat_frames)

        with gzip.open(fn_inf, 'wb') as fh:
            pickle.dump((beat_frames, jumps), fh)

    return y, sample_rate, beat_frames, jumps


def compute_buffers(y, beat_frames):
    int_max = numpy.iinfo(numpy.int16).max
    raw = (y * int_max).astype(numpy.int16).T.copy(order='C')

    buffers = []
    for start, end in iter_beat_slices(raw, beat_frames):
        buffers.append(y.T[start:end])

    return buffers


def normalize(jumps, threshold):
    n = len(jumps)

    jumps = enhance_diagonals(jumps, 0.8, 4)

    # scale
    x_max = jumps.max()
    x_min = x_max * threshold
    y_max = x_max ** 0.5
    jumps = (jumps - x_min) / (x_max - x_min) * y_max
    jumps *= jumps > 0

    # privilege jumps back in order to prolong playing
    jumps *= numpy.ones((n, n)) - numpy.tri(n, k=-1).T * 0.5

    # privilege wide jumps
    M = numpy.zeros((n, n))
    for i in range(1, n):
        M += numpy.tri(n, k=-i)
        M += numpy.tri(n, k=-i).T
    jumps *= (M / (n - 1)) ** 0.4

    return jumps


def get_next_position(i, jumps):
    for j, p in sorted(enumerate(jumps[i]), key=lambda jp: -jp[1]):
        if p > random():
            return j + 1
    return i + 1


def play(buffers, sample_rate, jumps):
    i = 0
    n = len(buffers)

    with soundcard.default_speaker().player(samplerate=sample_rate) as sp:
        while True:
            sp.play(buffers[i])
            print_progress(i, n)

            i = get_next_position(i, jumps)
            if i >= n:
                i = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.8, help='Between 0 and 1. '
        'A higher value will result in fewer but better jumps. (Default: 0.8)')
    parser.add_argument(
        '-f', '--force', action='store_true',
        help='Ignore previously saved analysis data.')
    return parser.parse_args()


def main():
    args = parse_args()

    print('Loading', args.filename)
    y, sample_rate, beat_frames, jumps = load(args.filename, args.force)
    jumps = normalize(jumps, args.threshold)
    buffers = compute_buffers(y, beat_frames)
    jump_count = sum(sum(jumps > 0))

    print('Detected {} jump opportunities on {} beats'.format(
        jump_count, len(buffers)
    ))

    print('Playing… (Press Ctrl-C to stop)')
    play(buffers, sample_rate, jumps)


if __name__ == '__main__':
    main()
