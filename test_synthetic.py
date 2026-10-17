"""Check the analyzer on a synthetic salsa-style groove (no audio hardware needed).

The groove has a cowbell on every beat, bass on 2-and / 4 / 6-and / 8,
conga open tones on 4 and 8, and a crash on 1 at the start of each phrase.
Run:  python test_synthetic.py
"""
import sys

import numpy as np

from analyzer import SR, SalsaAnalyzer


def tone(freq, dur, decay, amp):
    t = np.arange(int(dur * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)


def noise(dur, decay, amp, rng):
    t = np.arange(int(dur * SR)) / SR
    return amp * rng.standard_normal(t.size) * np.exp(-t / decay)


def make_groove(bpm, seconds, start, rng):
    audio = 0.01 * rng.standard_normal(int(seconds * SR))
    beat = 60.0 / bpm

    def add(t, sound):
        i = int(t * SR)
        if 0 <= i < audio.size:
            n = min(sound.size, audio.size - i)
            audio[i:i + n] += sound[:n]

    k = 0
    while start + k * beat < seconds:
        t = start + k * beat
        count = k % 8 + 1
        add(t, tone(3000, 0.05, 0.01, 0.25))                 # cowbell
        if count in (4, 8):
            add(t, tone(330, 0.25, 0.08, 0.6))               # conga open tone
            add(t, tone(70, 0.3, 0.12, 0.7))                 # bass
        if count in (2, 6):
            add(t + beat / 2, tone(70, 0.3, 0.12, 0.7))      # bass on the "and"
        if count == 1 and k % 16 == 0:
            add(t, noise(0.4, 0.15, 0.4, rng))               # crash on phrase start
        k += 1
    return audio.astype(np.float32)


def run(bpm, start=0.37, seconds=30.0, seed=0):
    rng = np.random.default_rng(seed)
    audio = make_groove(bpm, seconds, start, rng)
    an = SalsaAnalyzer()
    block = 1024
    for i in range(0, audio.size, block):
        an.feed(audio[i:i + block], i / SR)   # pretend t = 0 is the song start

    beat = 60.0 / bpm
    checked = correct = 0
    k = 0
    while start + (k + 0.5) * beat < seconds - 1:
        t = start + (k + 0.5) * beat       # middle of beat k
        if t > seconds * 0.6:              # only judge the second part
            checked += 1
            correct += an.count_at(t)["count"] == k % 8 + 1
        k += 1
    state = an.count_at(seconds - 1)
    ok = abs(state["bpm"] - bpm) / bpm < 0.02 and correct == checked
    print(f"{bpm:5.0f} BPM -> detected {state['bpm']:6.1f} BPM, "
          f"count right on {correct}/{checked} beats  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [run(bpm, seed=i) for i, bpm in enumerate([150, 170, 180, 195, 210])]
    sys.exit(0 if all(results) else 1)
