"Play an infinite remix of your favorite songs."

import argparse
import gzip
import pickle
import random
import shutil
from pathlib import Path

import librosa
import numpy
import scipy
import sklearn
import soundcard
from PIL import Image

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / 'timbre.pickle', 'rb') as fh:
    TIMBRE_PATTERNS = pickle.load(fh)


def colorize(k):
    colors = [31,32,34,93,95,96]
    return f"\033[{colors[k % len(colors)]}m{k}\033[0m"

class Progress:
    def __init__(self, n, segments):
        self.n = n
        self.indices = {}
        for k, v in segments.items():
            for beat in v:
                self.indices[beat] = colorize(k)

    def update(self, i):
        cols, _ = shutil.get_terminal_size()
        pos = lambda k: k * (cols - 7) // self.n
        s = (['='] * pos(i)) + (['-'] * (pos(self.n) - pos(i)))
        s[pos(i)] = '|'
        for x in self.indices:
            s[pos(x)] = self.indices[x]

        print(f'[{"".join(s)}] {i:>4}', end='\r')


def compute_buffers(y, beat_samples):
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


def get_track_segments(filename, sample_rate):
    # https://librosa.org/librosa_gallery/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py
    bins_per_octave = 12 * 3
    n_octaves = 7

    y_mono, _ = librosa.load(filename, sr=sample_rate)
    _, beats = librosa.beat.beat_track(y=y_mono, sr=sample_rate, trim=False)
    cqt = librosa.cqt(
            y=y_mono,
            sr=sample_rate,
            bins_per_octave=bins_per_octave,
            n_bins=n_octaves * bins_per_octave,
        )
    C = librosa.amplitude_to_db(numpy.abs(cqt), ref=numpy.max)
    Csync = librosa.util.sync(C, beats, aggregate=numpy.median)

    R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity', sym=True)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    mfcc = librosa.feature.mfcc(y=y_mono, sr=sample_rate)
    Msync = librosa.util.sync(mfcc, beats)

    path_distance = numpy.sum(numpy.diff(Msync, axis=1) ** 2, axis=0)
    sigma = numpy.median(path_distance)
    path_sim = numpy.exp(-path_distance / sigma)

    R_path = numpy.diag(path_sim, k=1) + numpy.diag(path_sim, k=-1)

    deg_path = numpy.sum(R_path, axis=1)
    deg_rec = numpy.sum(Rf, axis=1)

    mu = deg_path.dot(deg_path + deg_rec) / numpy.sum((deg_path + deg_rec) ** 2)

    A = mu * Rf + (1 - mu) * R_path

    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    _, evecs = scipy.linalg.eigh(L)

    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))

    Cnorm = numpy.cumsum(evecs**2, axis=1) ** 0.5

    k = 5

    X = evecs[:, :k] / Cnorm[:, k - 1 : k]

    KM = sklearn.cluster.KMeans(n_clusters=k)

    seg_ids = KM.fit_predict(X)

    bound_beats = 1 + numpy.flatnonzero(seg_ids[:-1] != seg_ids[1:])
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)
    bound_segs = list(seg_ids[bound_beats])

    segments = {}
    for label, beat in zip(bound_segs, bound_beats):
        if label not in segments:
            segments[label] = []
        segments[label].append(beat.item())

    discard = [label for label in segments if not segments[label]]
    for label in discard:
        del segments[label]

    ordered = sorted(segments.values(), key=lambda sublist: sorted(sublist))
    segments = dict(enumerate(ordered))

    return beats, segments


def jumps_from_segments(n, segments):
    jumps = numpy.eye(n)

    for frames in segments.values():
        jumps[numpy.ix_(frames, frames)] = 1.0

    return numpy.abs(jumps)


def load(filename, *, force=False):
    y, sample_rate = librosa.load(filename, mono=False, sr=None)

    path_inf = Path(filename + '.inf')
    if not force and path_inf.exists():
        with gzip.open(path_inf, 'rb') as fh:
            beat_frames, segments = pickle.load(fh)
    else:
        print('Analyzing…')
        beat_frames, segments = get_track_segments(filename, sample_rate)
        with gzip.open(path_inf, 'wb') as fh:
            pickle.dump((beat_frames, segments), fh)

    return compute_buffers(y, beat_frames), sample_rate, segments


def enhance(jumps, threshold):
    n = len(jumps)

    # beats are more similar if the surrounding beats are similar
    # for _ in range(4):
    #    jumps_before = numpy.roll(jumps, (-1, -1), (0, 1))
    #    jumps_after = numpy.roll(jumps, (1, 1), (0, 1))
    #    jumps = 0.4 * jumps_before + 0.4 * jumps_after + 0.2 * jumps

    # scale
    x_max = jumps.max()
    x_min = x_max * threshold
    y_max = x_max ** 0.5
    jumps = (jumps - x_min) / (x_max - x_min) * y_max
    jumps *= jumps > 0

    # privilege jumps back in order to prolong playing
    jumps[:] *= numpy.linspace(numpy.ones(n), numpy.ones(n) * 0.5, n)
    return jumps


def get_next_position(i, jumps, counts):
    n = len(jumps)
    j = numpy.array(range(n))
    w_count = (numpy.cumsum(counts[::-1] * (j + 1)) / numpy.cumsum(j + 1))[::-1]
    j = random.choices(range(n), jumps[i] / (w_count + 1))
    return j[0] + 1


def play(buffers, sample_rate, jumps, progress):
    i = 0
    n = len(buffers)
    counts = numpy.zeros(n)
    jumped = True  # never jump at start of playback

    with soundcard.default_speaker().player(samplerate=sample_rate, blocksize=8192) as sp:
        try:
            while True:
                progress.update(i)
                sp.play(buffers[i])
                counts[i] += 1

                if jumped:  # never jump two beats in a row
                    i += 1
                    jumped = False
                else:
                    j = get_next_position(i, jumps, counts)
                    jumped = j != (i + 1)
                    i = j
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
    buffers, sample_rate, segments = load(args.filename, force=args.force)
    progress = Progress(len(buffers), segments)
    jumps = jumps_from_segments(len(buffers), segments)
    jumps = enhance(jumps, args.threshold)
    jump_count = sum(sum(jumps > 0))

    print(f'Detected {jump_count} jump opportunities on {len(buffers)} beats')
    print('Playing… (Press Ctrl-C to stop)')
    play(buffers, sample_rate, jumps, progress)


if __name__ == '__main__':
    main()
