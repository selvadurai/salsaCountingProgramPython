"""Salsa Counter: listens to salsa and shows the 1-8 count.

  python app.py                 listen on the microphone
  python app.py song.mp3        play a song and count along
  python app.py --text          no window, print the count in the terminal
  python app.py --list-devices  show microphones / speakers

Keys (window): Space = "this beat is the 1", A = back to automatic 1,
O = switch On1 / On2, Left/Right = shift the display -/+ 10 ms, Q = quit.
"""
import argparse
import queue
import sys
import threading
import time

import numpy as np

try:
    import sounddevice as sd
except OSError:
    sys.exit("Audio library missing. Install it with:  sudo apt install libportaudio2")

from analyzer import SR, SalsaAnalyzer

BLOCK = 1024


# ------------------------------------------------------------------ audio in

def open_stream(kind, callback, device, **extra):
    """Open an input/output stream at SR, or at 2*SR if the device refuses SR."""
    cls = sd.InputStream if kind == "in" else sd.OutputStream
    for factor in (1, 2):
        try:
            return cls(samplerate=SR * factor, channels=1, dtype="float32",
                       blocksize=BLOCK * factor, device=device,
                       callback=callback(factor), **extra), factor
        except sd.PortAudioError:
            if factor == 2:
                raise


def downsample(block, factor):
    if factor == 1:
        return block
    n = len(block) // factor * factor
    return block[:n].reshape(-1, factor).mean(axis=1)


def start_mic(blocks, device):
    holder = {}

    def make_callback(factor):
        def callback(indata, frames, time_info, status):
            heard = time.monotonic() - holder["stream"].latency - frames / (SR * factor)
            blocks.put((downsample(indata[:, 0].copy(), factor), heard))
        return callback

    stream, _ = open_stream("in", make_callback, device)
    holder["stream"] = stream
    stream.start()
    return stream


def start_file(path, blocks, device, on_done):
    try:
        import librosa
    except ImportError:
        sys.exit("Playing files needs librosa:  pip install librosa")
    print(f"Loading {path} ...")
    holder = {"pos": 0}

    def make_callback(factor):
        audio, _ = librosa.load(path, sr=SR * factor, mono=True)
        holder["audio"] = audio.astype(np.float32)

        def callback(outdata, frames, time_info, status):
            pos = holder["pos"]
            chunk = holder["audio"][pos:pos + frames]
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):] = 0
            holder["pos"] = pos + frames
            if len(chunk):
                heard = time.monotonic() + holder["stream"].latency
                blocks.put((downsample(chunk.copy(), factor), heard))
            if len(chunk) < frames:
                raise sd.CallbackStop
        return callback

    stream, _ = open_stream("out", make_callback, device, finished_callback=on_done)
    holder["stream"] = stream
    stream.start()
    return stream


def analysis_loop(blocks, analyzer, stop):
    while not stop.is_set():
        try:
            block, heard = blocks.get(timeout=0.1)
        except queue.Empty:
            continue
        analyzer.feed(block, heard)


# ------------------------------------------------------------------- display

BREAKS = {"On1": (1, 5), "On2": (2, 6)}
PAUSES = {"On1": (4, 8), "On2": (1, 5)}

BG, FG, DIM, ACCENT, PAUSE = "#14101c", "#f4eee6", "#5b5468", "#ff7a3d", "#8c7fa8"


class CounterWindow:
    def __init__(self, analyzer, done):
        import tkinter as tk
        self.tk = tk
        self.analyzer, self.done = analyzer, done
        self.mode, self.sync_ms, self.flash = "On1", 0, ("", 0.0)

        self.root = tk.Tk()
        self.root.title("Salsa Counter")
        self.root.configure(bg=BG)
        self.root.geometry("640x520")
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.bind("<space>", lambda e: self.mark_one())
        self.root.bind("<KeyPress-a>", lambda e: self.auto())
        self.root.bind("<KeyPress-o>", lambda e: self.toggle_mode())
        self.root.bind("<Left>", lambda e: self.shift(-10))
        self.root.bind("<Right>", lambda e: self.shift(10))
        self.root.bind("<KeyPress-q>", lambda e: self.root.destroy())

    def now(self):
        return time.monotonic() + self.sync_ms / 1000.0

    def say(self, text):
        self.flash = (text, time.monotonic() + 1.5)

    def mark_one(self):
        self.say("1 set" if self.analyzer.mark_one(self.now()) else "Wait for the beat first")

    def auto(self):
        self.analyzer.set_auto()
        self.say("Finding the 1 automatically")

    def toggle_mode(self):
        self.mode = "On2" if self.mode == "On1" else "On1"
        self.say(f"Dancing {self.mode}")

    def shift(self, ms):
        self.sync_ms += ms
        self.say(f"Display shift {self.sync_ms:+d} ms")

    def run(self):
        self.tick()
        self.root.mainloop()

    def tick(self):
        self.draw(self.analyzer.count_at(self.now()))
        self.root.after(15, self.tick)

    def draw(self, s):
        c = self.canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 200)
        cx, cy = w / 2, h * 0.45

        if self.done.is_set():
            status = "Song finished"
        elif s["count"] is None:
            status = "Listening for salsa..."
        else:
            one = "1 set by you" if s["manual"] else ("1 found" if s["one_known"] else "finding the 1...")
            lock = "" if s["confident"] else "  (unsure)"
            status = f"{s['bpm']:.0f} BPM  ·  {one}  ·  {self.mode}{lock}"
        c.create_text(cx, 30, text=status, fill=FG, font=("Helvetica", 14))

        count = s["count"]
        if count is not None:
            color = ACCENT if count in BREAKS[self.mode] else PAUSE if count in PAUSES[self.mode] else FG
            if not s["confident"]:
                color = DIM
            pulse = (1 - s["phase"]) ** 3
            r = min(w, h) * (0.26 + 0.06 * pulse)
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=2 + 6 * pulse)
            c.create_text(cx, cy, text=str(count), fill=color,
                          font=("Helvetica", int(min(w, h) * 0.3), "bold"))

            gap = min(w / 10, 60)
            for i in range(1, 9):
                x = cx + (i - 4.5) * gap
                y = h * 0.82
                on = i == count
                fill = (ACCENT if i in BREAKS[self.mode] else FG) if on else DIM
                rr = 12 if on else 7
                c.create_oval(x - rr, y - rr, x + rr, y + rr, fill=fill, outline="")
                c.create_text(x, y + 26, text=str(i), fill=FG if on else DIM, font=("Helvetica", 11))
        else:
            c.create_text(cx, cy, text="♪", fill=DIM, font=("Helvetica", int(min(w, h) * 0.25)))

        text, until = self.flash
        if text and time.monotonic() < until:
            c.create_text(cx, h * 0.68, text=text, fill=ACCENT, font=("Helvetica", 16, "bold"))
        c.create_text(cx, h - 16, fill=DIM, font=("Helvetica", 10),
                      text="Space: this is the 1   A: auto   O: On1/On2   ←/→: sync   Q: quit")


def run_text(analyzer, done):
    print("Counting... (Ctrl+C to stop)")
    last = None
    try:
        while not done.is_set():
            s = analyzer.count_at(time.monotonic())
            if s["count"] != last:
                last = s["count"]
                if last is None:
                    print("\rListening for salsa...      ", end="", flush=True)
                else:
                    sure = "" if s["confident"] else " ?"
                    print(f"\r  {last}   ({s['bpm']:.0f} BPM){sure}      ", end="", flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    print()


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Shows the salsa 1-8 count of the music it hears.")
    p.add_argument("file", nargs="?", help="audio file to play (default: listen on the microphone)")
    p.add_argument("--text", action="store_true", help="print the count in the terminal instead of a window")
    p.add_argument("--device", help="input/output device number or name (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    analyzer = SalsaAnalyzer()
    blocks, stop, done = queue.Queue(), threading.Event(), threading.Event()
    threading.Thread(target=analysis_loop, args=(blocks, analyzer, stop), daemon=True).start()

    if args.file:
        stream = start_file(args.file, blocks, device, done.set)
    else:
        stream = start_mic(blocks, device)

    try:
        if args.text:
            run_text(analyzer, done)
        else:
            try:
                window = CounterWindow(analyzer, done)
            except ImportError:
                sys.exit("No window support. Install it with:  sudo apt install python3-tk\n"
                         "or run with --text")
            window.run()
    finally:
        stop.set()
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
