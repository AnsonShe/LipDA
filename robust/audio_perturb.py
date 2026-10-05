"""Audio perturbations for robustness tests."""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


class AudioPerturbation:
    def __init__(self):
        self.y = None
        self.sr = None
        self.audio_path = None

    def load_audio(self, audio_path: str, sr: Optional[int] = None):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=sr)

    def load_array(self, y: np.ndarray, sr: int):
        self.y = y
        self.sr = sr

    def add_noise(self, noise_factor: float = 0.01) -> np.ndarray:
        noise = np.random.normal(0, noise_factor, len(self.y))
        return self.y + noise

    def insert_silence(self, silence_duration: float = 0.5, position: Optional[float] = None) -> np.ndarray:
        silence_samples = int(silence_duration * self.sr)
        audio = self.y.copy()
        if position is None:
            position = float(np.random.uniform(0, max(len(audio) / self.sr, 1e-6)))
        insert_pos = int(position * self.sr)
        insert_pos = min(max(insert_pos, 0), len(audio))
        return np.concatenate([audio[:insert_pos], np.zeros(silence_samples), audio[insert_pos:]])

    def time_shift(self, shift_amount: float) -> np.ndarray:
        shift_samples = int(shift_amount * self.sr)
        if shift_samples > 0:
            return np.pad(self.y, (shift_samples, 0), mode="constant")
        return np.pad(self.y, (0, -shift_samples), mode="constant")

    def pitch_shift(self, n_steps: int) -> np.ndarray:
        return librosa.effects.pitch_shift(self.y, sr=self.sr, n_steps=n_steps)

    def time_stretch(self, rate: float) -> np.ndarray:
        return librosa.effects.time_stretch(self.y, rate=rate)

    def room_impulse_response(self, rir_path: Optional[str] = None, decay_time: float = 0.5) -> np.ndarray:
        if rir_path and os.path.exists(rir_path):
            rir, _ = librosa.load(rir_path, sr=self.sr)
        else:
            rir_length = max(int(decay_time * self.sr), 1)
            rir = np.random.normal(0, 0.01, rir_length)
            rir *= np.exp(-np.arange(rir_length) / (self.sr * decay_time))
        peak = np.max(np.abs(rir)) or 1.0
        rir = rir / peak
        convolved = signal.fftconvolve(self.y, rir, mode="same")
        src_peak = np.max(np.abs(self.y)) or 1.0
        dst_peak = np.max(np.abs(convolved)) or 1.0
        return convolved / dst_peak * src_peak

    def resample(self, target_sr: int) -> Tuple[np.ndarray, int]:
        return librosa.resample(self.y, orig_sr=self.sr, target_sr=target_sr), target_sr

    def apply_multiple(self, perturbations: dict) -> np.ndarray:
        audio = self.y.copy()
        saved = self.y
        try:
            if "noise" in perturbations:
                self.y = audio
                audio = self.add_noise(perturbations["noise"])
            if "silence" in perturbations:
                self.y = audio
                kwargs = perturbations["silence"]
                audio = self.insert_silence(**kwargs) if isinstance(kwargs, dict) else self.insert_silence(kwargs)
            if "time_shift" in perturbations:
                self.y = audio
                audio = self.time_shift(perturbations["time_shift"])
            if "pitch_shift" in perturbations:
                self.y = audio
                audio = self.pitch_shift(perturbations["pitch_shift"])
            if "time_stretch" in perturbations:
                self.y = audio
                audio = self.time_stretch(perturbations["time_stretch"])
            if "rir" in perturbations:
                self.y = audio
                kwargs = perturbations["rir"]
                audio = self.room_impulse_response(**kwargs) if isinstance(kwargs, dict) else self.room_impulse_response()
        finally:
            self.y = saved
        return audio

    def save_audio(self, output_path: str, audio: np.ndarray, sr: Optional[int] = None):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        sf.write(output_path, audio, self.sr if sr is None else sr)


PAPER_PRESETS = [
    ("noise_light", lambda p: (p.add_noise(0.005), p.sr, "noise_light.wav")),
    ("noise_heavy", lambda p: (p.add_noise(0.02), p.sr, "noise_heavy.wav")),
    ("silence", lambda p: (p.insert_silence(0.5, None), p.sr, "silence_random.wav")),
    ("shift_fwd", lambda p: (p.time_shift(1.0), p.sr, "time_shift_forward.wav")),
    ("shift_back", lambda p: (p.time_shift(-1.0), p.sr, "time_shift_backward.wav")),
    ("pitch_up", lambda p: (p.pitch_shift(3), p.sr, "pitch_up3.wav")),
    ("pitch_down", lambda p: (p.pitch_shift(-3), p.sr, "pitch_down3.wav")),
    ("rir_small", lambda p: (p.room_impulse_response(decay_time=0.2), p.sr, "rir_small.wav")),
    ("rir_medium", lambda p: (p.room_impulse_response(decay_time=0.5), p.sr, "rir_medium.wav")),
    ("rir_large", lambda p: (p.room_impulse_response(decay_time=1.0), p.sr, "rir_large.wav")),
    ("resample_16k", lambda p: (*p.resample(16000), "resample_16k.wav")),
    ("resample_44k", lambda p: (*p.resample(44100), "resample_44k.wav")),
]


def run_presets(perturb: AudioPerturbation, out_dir: str, names=None):
    os.makedirs(out_dir, exist_ok=True)
    wanted = set(names) if names else {row[0] for row in PAPER_PRESETS}
    written = []
    for key, fn in PAPER_PRESETS:
        if key not in wanted:
            continue
        audio, sr, fname = fn(perturb)
        path = os.path.join(out_dir, fname)
        perturb.save_audio(path, audio, sr)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Apply audio perturbations to a wav file")
    parser.add_argument("--in_wav", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--preset", default="all", help="all, or comma-separated keys from PAPER_PRESETS")
    args = parser.parse_args()
    perturb = AudioPerturbation()
    perturb.load_audio(args.in_wav)
    names = None if args.preset == "all" else [x.strip() for x in args.preset.split(",") if x.strip()]
    paths = run_presets(perturb, args.out_dir, names)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
