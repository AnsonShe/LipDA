import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import os
from typing import Tuple, Optional


class AudioPerturbation:
    """Audio perturbation class for applying various audio distortions"""

    def __init__(self):
        ...
        
    def load_audio(self, audio_path: str):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=None)
        self.original_sr = self.sr

    def add_noise(self, noise_factor: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to audio"""
        noise = np.random.normal(0, noise_factor, len(self.y))
        return self.y + noise

    def insert_silence(
        self, silence_duration: float = 0.5, position: Optional[float] = None
    ) -> np.ndarray:
        """
        Insert silence into audio

        Args:
            silence_duration: Duration of silence in seconds
            position: Position to insert (seconds). If None, random position
        """
        silence_samples = int(silence_duration * self.sr)
        audio = self.y.copy()

        if position is None:
            position = np.random.uniform(0, len(audio) / self.sr)

        insert_pos = int(position * self.sr)
        silence = np.zeros(silence_samples)

        perturbed = np.concatenate([audio[:insert_pos], silence, audio[insert_pos:]])
        return perturbed

    def time_shift(self, shift_amount: float) -> np.ndarray:
        """
        Time shift audio

        Args:
            shift_amount: Shift amount in seconds (positive or negative)
        """
        shift_samples = int(shift_amount * self.sr)

        if shift_samples > 0:
            return np.pad(self.y, (shift_samples, 0), mode="constant")
        else:
            return np.pad(self.y, (0, -shift_samples), mode="constant")

    def pitch_shift(self, n_steps: int) -> np.ndarray:
        """
        Shift pitch by semitones

        Args:
            n_steps: Number of semitones to shift (positive or negative)
        """
        return librosa.effects.pitch_shift(self.y, sr=self.sr, n_steps=n_steps)

    def time_stretch(self, rate: float) -> np.ndarray:
        """
        Time stretch audio (changes speed without changing pitch)

        Args:
            rate: Stretch rate (> 1 slower, < 1 faster)
        """
        return librosa.effects.time_stretch(self.y, rate=rate)

    def room_impulse_response(
        self, rir_path: Optional[str] = None, decay_time: float = 0.5
    ) -> np.ndarray:
        """
        Apply room impulse response convolution

        Args:
            rir_path: Path to RIR file. If None, generate synthetic RIR
            decay_time: Decay time in seconds (for synthetic RIR)
        """
        if rir_path and os.path.exists(rir_path):
            rir, _ = librosa.load(rir_path, sr=self.sr)
        else:
            # Generate synthetic RIR
            rir_length = int(decay_time * self.sr)
            rir = np.random.normal(0, 0.01, rir_length)
            rir *= np.exp(-np.arange(rir_length) / (self.sr * decay_time))

        # Normalize RIR
        rir = rir / np.max(np.abs(rir))

        # Convolve
        convolved = signal.fftconvolve(self.y, rir, mode="same")

        # Normalize to prevent clipping
        convolved = convolved / np.max(np.abs(convolved)) * np.max(np.abs(self.y))

        return convolved

    def resample(self, target_sr: int) -> Tuple[np.ndarray, int]:
        """
        Resample audio

        Args:
            target_sr: Target sample rate

        Returns:
            Tuple of (resampled_audio, new_sample_rate)
        """
        resampled = librosa.resample(self.y, orig_sr=self.sr, target_sr=target_sr)
        return resampled, target_sr

    def apply_multiple_perturbations(self, perturbations: dict) -> np.ndarray:
        """
        Apply multiple perturbations sequentially

        Args:
            perturbations: Dict with keys like 'noise', 'silence', 'pitch_shift', etc.
                          and their respective parameters
        """
        audio = self.y.copy()

        if "noise" in perturbations:
            temp_y = self.y
            self.y = audio
            audio = self.add_noise(perturbations["noise"])
            self.y = temp_y

        if "silence" in perturbations:
            temp_y = self.y
            self.y = audio
            audio = self.insert_silence(**perturbations["silence"])
            self.y = temp_y

        if "time_shift" in perturbations:
            temp_y = self.y
            self.y = audio
            audio = self.time_shift(perturbations["time_shift"])
            self.y = temp_y

        if "pitch_shift" in perturbations:
            temp_y = self.y
            self.y = audio
            audio = self.pitch_shift(perturbations["pitch_shift"])
            self.y = temp_y

        if "time_stretch" in perturbations:
            temp_y = self.y
            self.y = audio
            audio = self.time_stretch(perturbations["time_stretch"])
            self.y = temp_y

        if "rir" in perturbations:
            temp_y = self.y
            self.y = audio
            audio = self.room_impulse_response(**perturbations["rir"])
            self.y = temp_y

        return audio

    def save_audio(self, output_path: str, audio: np.ndarray, sr: Optional[int] = None):
        """Save perturbed audio to file"""
        if sr is None:
            sr = self.sr
        sf.write(output_path, audio, sr)
        # print(f"Saved to {output_path}")


# ...existing code...

# Example usage
if __name__ == "__main__":
    # Initialize with your audio file
    audio_file = "/Users/aaroncomo/Aaron/创作/Call Of Silence/demo3.wav"

    perturb = AudioPerturbation()
    perturb.load_audio(audio_file)

    output_dir = "/Users/aaroncomo/coding/python/output_wav"
    
    print("=" * 50)
    print("开始音频扰动测试")
    print("=" * 50)

    # Example 1: Add noise (添加噪声)
    print("\n[1] 测试: 添加高斯噪声")
    noisy_audio = perturb.add_noise(noise_factor=0.005)
    perturb.save_audio(f"{output_dir}/output_noise_light.wav", noisy_audio)
    
    noisy_audio_heavy = perturb.add_noise(noise_factor=0.02)
    perturb.save_audio(f"{output_dir}/output_noise_heavy.wav", noisy_audio_heavy)

    # Example 2: Insert silence (插入静音)
    print("\n[2] 测试: 插入静音")
    silence_audio = perturb.insert_silence(silence_duration=0.5, position=None)
    perturb.save_audio(f"{output_dir}/output_silence_random.wav", silence_audio)
    

    # Example 3: Time shift (时间偏移)
    print("\n[3] 测试: 时间偏移")
    shifted_audio_forward = perturb.time_shift(shift_amount=1)
    perturb.save_audio(f"{output_dir}/output_time_shift_forward.wav", shifted_audio_forward)
    
    shifted_audio_backward = perturb.time_shift(shift_amount=-1)
    perturb.save_audio(f"{output_dir}/output_time_shift_backward.wav", shifted_audio_backward)

    # Example 4: Pitch shift (音高变换)
    print("\n[4] 测试: 音高变换")
    pitched_audio_up = perturb.pitch_shift(n_steps=3)
    perturb.save_audio(f"{output_dir}/output_pitch_shift_up3.wav", pitched_audio_up)
    
    pitched_audio_down = perturb.pitch_shift(n_steps=-3)
    perturb.save_audio(f"{output_dir}/output_pitch_shift_down3.wav", pitched_audio_down)


    # Example 6: Room impulse response (房间脉冲响应卷积)
    print("\n[6] 测试: 房间脉冲响应卷积")
    rir_audio_small = perturb.room_impulse_response(decay_time=0.2)
    perturb.save_audio(f"{output_dir}/output_rir_small_room.wav", rir_audio_small)
    
    rir_audio_medium = perturb.room_impulse_response(decay_time=0.5)
    perturb.save_audio(f"{output_dir}/output_rir_medium_room.wav", rir_audio_medium)
    
    rir_audio_large = perturb.room_impulse_response(decay_time=1.0)
    perturb.save_audio(f"{output_dir}/output_rir_large_room.wav", rir_audio_large)

    # Example 7: Resample (重采样)
    print("\n[7] 测试: 重采样")
    resampled_audio_16k, sr_16k = perturb.resample(target_sr=16000)
    perturb.save_audio(f"{output_dir}/output_resample_16k.wav", resampled_audio_16k, sr_16k)
    
    
    resampled_audio_44k, sr_44k = perturb.resample(target_sr=44100)
    perturb.save_audio(f"{output_dir}/output_resample_44k.wav", resampled_audio_44k, sr_44k)

    print("\n" + "=" * 50)
    print("✓ 所有扰动测试完成！")
    print("=" * 50)
    print(f"\n生成的文件位置: {output_dir}")
    print("\n生成的文件列表:")
    print("  1. output_noise_light.wav - 轻度噪声")
    print("  2. output_noise_heavy.wav - 重度噪声")
    print("  3. output_silence_random.wav - 随机位置插入静音")
    print("  5. output_time_shift_forward.wav - 向前时间偏移")
    print("  6. output_time_shift_backward.wav - 向后时间偏移")
    print("  7. output_pitch_shift_up3.wav - 升高3个半音")
    print("  8. output_pitch_shift_down3.wav - 降低3个半音")
    print(" 12. output_rir_small_room.wav - 小房间脉冲")
    print(" 13. output_rir_medium_room.wav - 中等房间脉冲")
    print(" 14. output_rir_large_room.wav - 大房间脉冲")
    print(" 15. output_resample_16k.wav - 重采样至16kHz")
    print(" 17. output_resample_44k.wav - 重采样至44.1kHz")
