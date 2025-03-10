"Play an infinite remix of your favorite songs."

import argparse
import gzip
import pickle
import random
import shutil
from pathlib import Path

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
        y_mono, _ = librosa.load(filename)
        tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sample_rate)
        buffers_mono = compute_buffers(y_mono, beat_frames)
        jumps = analyze(buffers_mono)

        with gzip.open(path_inf, 'wb') as fh:
            pickle.dump((beat_frames, jumps), fh)

    return compute_buffers(y, beat_frames), sample_rate, jumps


def enhance(jumps, threshold):
    n = len(jumps)

    # beats are more similar if the surrounding beats are similar
    for _ in range(4):
        jumps_before = numpy.roll(jumps, (-1, -1), (0, 1))
        jumps_after = numpy.roll(jumps, (1, 1), (0, 1))
        jumps = 0.4 * jumps_before + 0.4 * jumps_after + 0.2 * jumps

    # scale
    x_max = jumps.max()
    x_min = x_max * threshold
    y_max = x_max ** 0.5
    jumps = (jumps - x_min) / (x_max - x_min) * y_max
    jumps *= jumps > 0
    jumps += numpy.eye(n)

    # privilege jumps back in order to prolong playing
    jumps[:] *= numpy.linspace(numpy.ones(n), numpy.ones(n) * 0.5, n)

    return jumps


def get_next_position(i, jumps, counts):
    n = len(jumps)
    j = numpy.array(range(n))
    w_count = (numpy.cumsum(counts[::-1] * (j + 1)) / numpy.cumsum(j + 1))[::-1]
    j = random.choices(range(n), jumps[i] / (w_count + 1))
    return j[0] + 1


def play(buffers, sample_rate, jumps):
    i = 0
    n = len(buffers)
    counts = numpy.zeros(n)

    with soundcard.default_speaker().player(samplerate=sample_rate) as sp:
        try:
            while True:
                sp.play(buffers[i])
                counts[i] += 1
                print_progress(i, n)

                i = get_next_position(i, jumps, counts)
                if i >= n:
                    i = 0
        except KeyboardInterrupt:
            print('\nStopping…')


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
    jumps = enhance(jumps, args.threshold)
    jump_count = sum(sum(jumps > 0))

    print(f'Detected {jump_count} jump opportunities on {len(buffers)} beats')
    print('Playing… (Press Ctrl-C to stop)')
    play(buffers, sample_rate, jumps)


if __name__ == '__main__':
    main()
