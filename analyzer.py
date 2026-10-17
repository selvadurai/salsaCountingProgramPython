"""Real-time salsa beat tracking and 1-8 count estimation.

Feed mono audio blocks (at SR) together with the time.monotonic() moment
their first sample is heard, then ask count_at(t) for the count at any
moment. Only numpy is required, so this module can be tested without audio
hardware.

How it works:
  1. Onset envelope: spectral flux (how much the spectrum suddenly gets
     louder) for the full band and for a low band (bass + congas).
  2. Tempo: autocorrelation of the last few seconds of onsets, limited to
     the salsa range, and refined to sub-frame precision.
  3. Phase: slide a comb of beats over the envelope to find where beats land.
  4. A "beat clock" smooths tempo and phase so the count does not jitter.
  5. Finding the 1: in salsa the bass and the conga open tones land on 4
     and 8, so the beats with the strongest low-band accents are taken to be
     4 and 8. Crashes and horn hits tend to start phrases, so the stronger
     full-band accent decides between 1 and 5. This is a heuristic, and the
     user can always override it.
"""
import math
import threading
from collections import deque

import numpy as np

SR = 22050
HOP = 256
N_FFT = 1024

TEMPO_EVERY = 16          # frames between tempo/phase updates (~0.19 s)
MIN_HISTORY_S = 3.0       # seconds of audio before the first estimate
BEAT_DECAY = 0.985        # memory of per-beat accents (~70 beats)
SWITCH_VOTES = 8          # beats a new "1" must win before we switch to it


class SalsaAnalyzer:
    def __init__(self, sr=SR, min_bpm=140.0, max_bpm=240.0, window_s=8.0):
        self.sr = sr
        self.fps = sr / HOP
        self.min_bpm, self.max_bpm = min_bpm, max_bpm
        self.max_frames = int(window_s * self.fps)
        self.lock = threading.Lock()

        self._window = np.hanning(N_FFT).astype(np.float32)
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
        self._full_bins = (freqs >= 40) & (freqs <= 8000)
        self._low_bins = (freqs >= 40) & (freqs <= 400)

        self._pending = np.zeros(0, np.float32)
        self._pending_t = None
        self._prev_spec = None
        self._env_full = np.zeros(0)
        self._env_low = np.zeros(0)
        self._times = np.zeros(0)
        self._rms = deque(maxlen=int(2 * sr / 1024) + 1)
        self._frames_since_tempo = 0

        self.offset = None        # beat index (mod 8) that is count 1
        self.manual = False       # True once the user marked the 1
        self.reset_clock()

    # ------------------------------------------------------------------ input

    def feed(self, samples, t_start):
        """Add a block of mono float samples whose first sample is heard at t_start."""
        samples = np.asarray(samples, np.float32)
        if samples.size == 0:
            return
        with self.lock:
            self._rms.append(float(np.sqrt(np.mean(samples ** 2))))
            if self._pending_t is None:
                self._pending_t = t_start
            else:
                # Trust the sample count, but slowly follow the wall clock
                # and snap after a dropout.
                expected = self._pending_t + len(self._pending) / self.sr
                drift = t_start - expected
                self._pending_t += drift if abs(drift) > 0.1 else 0.02 * drift
            self._pending = np.concatenate([self._pending, samples])
            self._process_frames()

    def _process_frames(self):
        full, low, times = [], [], []
        while len(self._pending) >= N_FFT:
            frame = self._pending[:N_FFT] * self._window
            spec = np.log1p(100.0 * np.abs(np.fft.rfft(frame)))
            if self._prev_spec is None:
                self._prev_spec = spec
            flux = np.maximum(spec - self._prev_spec, 0.0)
            self._prev_spec = spec
            full.append(flux[self._full_bins].mean())
            low.append(flux[self._low_bins].mean())
            times.append(self._pending_t + 0.5 * N_FFT / self.sr)
            self._pending = self._pending[HOP:]
            self._pending_t += HOP / self.sr
        if not full:
            return

        keep = self.max_frames
        self._env_full = np.concatenate([self._env_full, full])[-keep:]
        self._env_low = np.concatenate([self._env_low, low])[-keep:]
        self._times = np.concatenate([self._times, times])[-keep:]

        self._frames_since_tempo += len(full)
        if self._frames_since_tempo >= TEMPO_EVERY:
            self._frames_since_tempo = 0
            self._update_tempo()
        self._collect_beat_accents()

    # ------------------------------------------------------------ beat clock

    def reset_clock(self):
        self.period = None        # seconds per beat
        self.anchor_t = 0.0
        self.anchor_idx = 0.0
        self.confidence = 0.0
        self._disagree = 0
        self._next_accent_beat = None
        self._acc_low = np.zeros(8)
        self._acc_full = np.zeros(8)
        self._acc_w = np.zeros(8)
        self._recent_low = deque(maxlen=32)
        self._recent_full = deque(maxlen=32)
        self._beats_seen = 0
        self._votes = 0
        if not self.manual:
            self.offset = None

    def _idx_at(self, t):
        return self.anchor_idx + (t - self.anchor_t) / self.period

    def _time_of(self, idx):
        return self.anchor_t + (idx - self.anchor_idx) * self.period

    def _update_tempo(self):
        if self._rms and max(self._rms) < 1e-3:
            # Silence: forget the last song so it can't confuse the next one.
            self.reset_clock()
            self._env_full, self._env_low, self._times = self._env_full[:0], self._env_low[:0], self._times[:0]
            return
        est = self._estimate_beat()
        if est is None:
            return
        beat_t, period, conf = est
        self.confidence = 0.8 * self.confidence + 0.2 * conf

        if self.period is None:
            if conf < 0.1:
                return
            self.period, self.anchor_t, self.anchor_idx = period, beat_t, 0.0
            # Back-fill accents from the audio we already have.
            self._next_accent_beat = math.ceil(self._idx_at(self._times[0] + 0.06))
            return

        if abs(period / self.period - 1.0) > 0.06:
            self._disagree += 1
            if self._disagree >= 5:  # tempo really changed: start over
                # Keep the beat numbering so a manually marked 1 survives.
                idx = float(round(self._idx_at(beat_t)))
                self.reset_clock()
                self.period, self.anchor_t, self.anchor_idx = period, beat_t, idx
                self._next_accent_beat = math.ceil(self._idx_at(self._times[0] + 0.06))
            return
        self._disagree = 0

        # Nudge tempo and phase towards the measurement (a simple PLL), keeping
        # the integer beat numbering continuous.
        idx = self._idx_at(beat_t)
        err = idx - round(idx)
        self.period += 0.1 * (period - self.period)
        self.anchor_t = beat_t
        self.anchor_idx = round(idx) + 0.7 * err

    def _estimate_beat(self):
        n = len(self._env_full)
        if n < MIN_HISTORY_S * self.fps:
            return None
        env = self._env_full + self._env_low
        env = np.convolve(env, [0.25, 0.5, 0.25], mode="same")

        # Tempo: autocorrelation. A real beat period also lines up with every
        # multiple of itself, which tells it apart from the off-beat bass and
        # 3-against-4 patterns that fool a single-lag peak.
        e = env - env.mean()
        spec = np.fft.rfft(e, 2 * n)
        ac = np.fft.irfft(spec * np.conj(spec))[:n]
        if ac[0] <= 1e-12:
            return None
        ac /= ac[0]
        lags = np.arange(60 * self.fps / self.max_bpm, 60 * self.fps / self.min_bpm, 0.1)
        m = np.arange(1, 9)
        at = lags[:, None] * m[None, :]
        score = np.where(at < n / 2, np.interp(at, np.arange(n), ac), np.nan)
        score = np.nanmean(score, axis=1)
        period_frames = float(lags[int(np.nanargmax(score))])
        conf = float(max(np.nanmax(score), 0.0))

        # Phase: comb over the envelope, recent beats weighted more.
        k = np.arange(int((n - 1) / period_frames))
        weights = 0.9 ** k
        phis = np.arange(0.0, period_frames, 0.25)
        pos = (n - 1) - phis[:, None] - k[None, :] * period_frames
        vals = np.interp(pos, np.arange(n), env, left=0.0)
        best_phi = phis[int(np.argmax((vals * weights).sum(axis=1)))]

        beat_frame = (n - 1) - best_phi
        beat_t = float(np.interp(beat_frame, np.arange(n), self._times))
        return beat_t, period_frames / self.fps, conf

    # ------------------------------------------------------- finding the "1"

    def _collect_beat_accents(self):
        if self.period is None or self._next_accent_beat is None:
            return
        latest = self._times[-1]
        while True:
            tb = self._time_of(self._next_accent_beat)
            if tb + 0.08 > latest:
                break
            sel = (self._times >= tb - 0.05) & (self._times <= tb + 0.05)
            if sel.any():
                self._add_beat(self._next_accent_beat,
                               self._env_low[sel].max(), self._env_full[sel].max())
            self._next_accent_beat += 1

    def _add_beat(self, beat, low, full):
        self._recent_low.append(low)
        self._recent_full.append(full)
        r = beat % 8
        for acc in (self._acc_low, self._acc_full, self._acc_w):
            acc *= BEAT_DECAY
        self._acc_low[r] += low / (np.median(self._recent_low) + 1e-9)
        self._acc_full[r] += full / (np.median(self._recent_full) + 1e-9)
        self._acc_w[r] += 1.0
        self._beats_seen += 1
        self._update_offset()

    def _update_offset(self):
        if self.manual or self._beats_seen < 16 or self._acc_w.min() == 0:
            return
        low = self._acc_low / self._acc_w
        full = self._acc_full / self._acc_w
        # Counts 4 and 8 sit 3 and 7 beats after the 1.
        o4 = int(np.argmax([low[(o + 3) % 8] + low[(o + 7) % 8] for o in range(4)]))
        candidate = o4 if full[o4] >= full[o4 + 4] else o4 + 4

        if self.offset is None:
            self.offset = candidate
        elif candidate != self.offset:
            self._votes += 1
            if self._votes >= SWITCH_VOTES:
                self.offset, self._votes = candidate, 0
        else:
            self._votes = 0

    # ---------------------------------------------------------------- output

    def mark_one(self, t):
        """The user says the beat nearest to time t is the 1."""
        with self.lock:
            if self.period is None:
                return False
            self.offset = int(round(self._idx_at(t))) % 8
            self.manual = True
            return True

    def set_auto(self):
        with self.lock:
            self.manual = False
            self._votes = 0
            self.offset = None
            self._update_offset()

    def count_at(self, t):
        """Return the state of the count at time t (time.monotonic() seconds)."""
        with self.lock:
            if self.period is None:
                return {"count": None, "phase": 0.0, "bpm": None, "confident": False,
                        "one_known": False, "manual": self.manual}
            idx = self._idx_at(t)
            beat = math.floor(idx)
            offset = self.offset if self.offset is not None else 0
            return {
                "count": (beat - offset) % 8 + 1,
                "phase": idx - beat,
                "bpm": 60.0 / self.period,
                "confident": self.confidence > 0.15,
                "one_known": self.offset is not None,
                "manual": self.manual,
            }
