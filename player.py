"Play an infinite remix of your favorite songs."

import argparse
import gzip
import pickle
import shutil
from pathlib import Path
from random import random

import librosa
import numpy
import soundcard
from PIL import Image

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / 'timbre.pickle', 'rb') as fh:
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
    for _ in range(steps):
        # combine each cell with its diagonal neighbors
        jumps1 = numpy.roll(jumps, (1, 1), (0, 1))
        jumps2 = numpy.roll(jumps, (-1, -1), (0, 1))
        jumps = (weight * (jumps1 + jumps2) + (1 - weight) * jumps) / 2
    return jumps


def compute_buffers(y, beat_frames):
    beat_samples = librosa.frames_to_samples(beat_frames)
    ranges = zip([0, *beat_samples], [*beat_samples, None])
    return [y.T[start:end] for start, end in ranges]


def timbre(y):
    spectrum = numpy.abs(librosa.stft(y))
    resized = numpy.array(Image.fromarray(spectrum).resize((70, 50)))

    k = len(TIMBRE_PATTERNS)
    t = numpy.zeros((k, k))
    s = numpy.zeros((k, 1))

    for i, pattern in enumerate(TIMBRE_PATTERNS):
        s[i][0] = numpy.sum(TIMBRE_PATTERNS[i] * resized)
        for j, pattern2 in enumerate(TIMBRE_PATTERNS):
            t[i][j] = numpy.sum(pattern * pattern2)

    return numpy.linalg.inv(t) @ s


def analyze(buffers):
    timbres = numpy.array([timbre(buf) for buf in buffers]).T
    return librosa.segment.recurrence_matrix(timbres, width=4, mode='affinity')


def load(filename, *, force=False):
    y, sample_rate = librosa.load(filename, mono=False)

    path_inf = Path(filename + '.inf')
    if not force and path_inf.exists():
        with gzip.open(path_inf, 'rb') as fh:
            beat_frames, jumps = pickle.load(fh)
    else:
        print('Analyzing…')
        y1, sample_rate1 = librosa.load(filename)
        tempo, beat_frames = librosa.beat.beat_track(y=y1, sr=sample_rate1)
        buffers1 = compute_buffers(y1, beat_frames)
        jumps = analyze(buffers1)

        with gzip.open(path_inf, 'wb') as fh:
            pickle.dump((beat_frames, jumps), fh)

    return compute_buffers(y, beat_frames), sample_rate, jumps


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
    m = numpy.zeros((n, n))
    for i in range(1, n):
        m += numpy.tri(n, k=-i)
        m += numpy.tri(n, k=-i).T
    jumps *= (m / (n - 1)) ** 0.4

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
    buffers, sample_rate, jumps = load(args.filename, force=args.force)
    jumps = normalize(jumps, args.threshold)
    jump_count = sum(sum(jumps > 0))

    print(f'Detected {jump_count} jump opportunities on {len(buffers)} beats')
    print('Playing… (Press Ctrl-C to stop)')
    play(buffers, sample_rate, jumps)


if __name__ == '__main__':
    main()
