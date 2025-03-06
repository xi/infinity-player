# infinity player

An infinite jukebox clone using librosa.

This program attempts to recreate the wonderful *Infinite Jukebox* on the
command line in python.  It plays the song beat by beat and may jump to similar
beats at any time. This way, the song can play infinitly.

The process is devided into two steps: Analysing the audio and playing it.
Analysing takes some time, so the result is saved next to the audiofile.

## Quickstart

    pip install -r requirements.txt
    python player.py <filename>

## Open Tasks

Any help would be appreciated

-   **Improve audio analysis.** I don't really know what *MFCC* or a
    *Constant-Q chromagram* is. The current implementation works ok-ish, but
    there can probably be big improvements.

-   **Improve beat selection.** I guess a lot more is possible, e.g. to prevent
    the song from looping through the same part again and again.

## Prior Art

-   Of course, the now defunct original [Infinite
    Jukebox](http://labs.echonest.com/Uploader/)
-   A functional fork called [Eternal Jukebox](https://eternal.abimon.org/)
    ([code](https://github.com/UnderMybrella/EternalJukebox)). It uses the
    [audio-analysis](https://developer.spotify.com/web-api/get-audio-analysis/)
    endpoint at spotify
-   [Remixatron](https://github.com/drensin/Remixatron), a similar project to
    this one.
