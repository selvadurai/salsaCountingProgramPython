# Salsa Counter

Listens to salsa music and shows the dance count (1-8) on screen, in time with the music.

## Setup ( Python 3.10)

```bash
sudo apt install python3-venv python3-tk libportaudio2 ffmpeg
cd salsa-counter
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py                 # listen with the microphone
python app.py song.mp3        # play a song and count along (most accurate)
python app.py --text          # terminal only, no window
python app.py --list-devices  # pick another mic with --device N
```

It needs about 3-5 seconds of music to lock onto the beat, and a few more to find the 1.

| Key | What it does |
|-----|--------------|
| Space | "This beat is the 1": press it on a 1 if the app picked the wrong beat |
| A | Go back to finding the 1 automatically |
| O | Switch between On1 and On2 (changes which counts are highlighted as break steps) |
| ← / → | Shift the display 10 ms earlier/later if the numbers feel late or early |
| Q | Quit |

Colours: orange marks the break steps (1 & 5 On1, 2 & 6 On2), purple marks the pauses, and grey means the app isn't sure yet.

## How it works (`analyzer.py`)

1. **Onsets:** finds sudden increases in loudness, both across the full sound and in the bass/conga range.
2. **Tempo:** finds the repeating interval (salsa range 140-240 BPM). Checking that the interval also repeats at 2x, 3x, 4x... avoids being fooled by the off-beat bass.
3. **Beat clock:** smooths tempo and timing so the count stays steady, and predicts beats so the number appears on time instead of late.
4. **Finding the 1:** bass and conga open tones hit on **4 and 8**, so the most bass-heavy beats are taken as 4 and 8. Crashes and horn hits that start phrases decide between 1 and 5. This is the weakest step, which is why Space lets you correct it.

## Test

```bash
python test_synthetic.py
```
This runs the analyzer on generated salsa-style grooves (150-210 BPM) with no audio hardware.
