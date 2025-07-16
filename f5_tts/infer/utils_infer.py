# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
from concurrent.futures import ThreadPoolExecutor

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib

matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import snapshot_download, hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
)

_ref_audio_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.25
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------

# Silence token processing functions

def parse_silence_tokens(text):
    """
    Parse silence tokens in format <<<sil#duration>>> and extract silence information
    
    Args:
        text (str): Input text with silence tokens
        
    Returns:
        tuple: (cleaned_text, silence_segments)
            - cleaned_text: Text with silence tokens removed
            - silence_segments: List of (position, duration_ms) tuples
    """
    import re
    
    silence_pattern = r'<<<sil#(\d+)>>>'
    silence_segments = []
    
    # Find all silence tokens and their positions
    matches = list(re.finditer(silence_pattern, text))
    
    # Process matches in reverse order to maintain correct positions
    cleaned_text = text
    offset = 0
    
    for match in reversed(matches):
        duration_ms = int(match.group(1))
        start_pos = match.start() - offset
        end_pos = match.end() - offset
        
        # Store position in cleaned text and duration
        silence_segments.insert(0, (start_pos, duration_ms))
        
        # Remove the silence token from text
        cleaned_text = cleaned_text[:match.start()] + cleaned_text[match.end():]
        offset += len(match.group(0))
    
    return cleaned_text, silence_segments


def validate_silence_tokens(text):
    """
    Validate silence token placement to avoid issues
    
    Args:
        text (str): Input text with silence tokens
        
    Returns:
        tuple: (is_valid, error_message)
    """
    import re
    
    silence_pattern = r'<<<sil#(\d+)>>>'
    
    # Check for tokens without proper spacing
    invalid_patterns = [
        r'\S<<<sil#\d+>>>',  # No space before token
        r'<<<sil#\d+>>>\S',  # No space after token
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, text):
            return False, "Silence tokens must be surrounded by spaces"
    
    # Check for valid duration values
    matches = re.findall(r'<<<sil#(\d+)>>>', text)
    for duration in matches:
        duration_ms = int(duration)
        if duration_ms < 50 or duration_ms > 5000:
            return False, f"Silence duration {duration_ms}ms is out of range (50-5000ms)"
    
    return True, ""


def apply_silence_to_audio(audio_wave, silence_segments, sample_rate=24000, text_length=None):
    """
    Apply silence segments to generated audio based on text positions
    
    Args:
        audio_wave (numpy.ndarray): Generated audio wave
        silence_segments (list): List of (text_position, duration_ms) tuples
        sample_rate (int): Audio sample rate
        text_length (int): Length of the cleaned text for better position calculation
        
    Returns:
        numpy.ndarray: Audio with silence applied
    """
    if not silence_segments:
        return audio_wave
    
    # Simple approach: insert silence at proportional positions
    result_segments = []
    audio_length = len(audio_wave)
    
    # Sort silence segments by position
    sorted_segments = sorted(silence_segments, key=lambda x: x[0])
    
    current_audio_pos = 0
    
    for text_pos, duration_ms in sorted_segments:
        # Improved proportional mapping using actual text length
        if text_length and text_length > 0:
            relative_pos = min(text_pos / text_length, 1.0)
        else:
            # Fallback to rough estimation
            relative_pos = min(text_pos / 100.0, 1.0)
        
        audio_pos = int(relative_pos * audio_length)
        audio_pos = max(current_audio_pos, min(audio_pos, audio_length))
        
        # Add audio segment before silence
        if audio_pos > current_audio_pos:
            result_segments.append(audio_wave[current_audio_pos:audio_pos])
        
        # Add silence
        silence_samples = int(duration_ms * sample_rate / 1000)
        silence = np.zeros(silence_samples, dtype=audio_wave.dtype)
        result_segments.append(silence)
        
        current_audio_pos = audio_pos
    
    # Add remaining audio
    if current_audio_pos < audio_length:
        result_segments.append(audio_wave[current_audio_pos:])
    
    return np.concatenate(result_segments) if result_segments else audio_wave


def insert_silence_in_audio(audio_segments, silence_segments, sample_rate=24000):
    """
    Insert silence into audio based on silence segments (legacy function)
    
    Args:
        audio_segments (list): List of audio numpy arrays
        silence_segments (list): List of (position, duration_ms) tuples
        sample_rate (int): Audio sample rate
        
    Returns:
        numpy.ndarray: Audio with silence inserted
    """
    if not silence_segments:
        return np.concatenate(audio_segments) if audio_segments else np.array([])
    
    # Convert duration to samples
    silence_samples = []
    for pos, duration_ms in silence_segments:
        samples = int(duration_ms * sample_rate / 1000)
        silence_samples.append((pos, samples))
    
    # Insert silence at appropriate positions
    result_audio = []
    current_pos = 0
    
    for i, audio_segment in enumerate(audio_segments):
        # Check if we need to insert silence before this segment
        for pos, samples in silence_samples:
            if pos == current_pos:
                silence = np.zeros(samples, dtype=audio_segment.dtype)
                result_audio.append(silence)
        
        result_audio.append(audio_segment)
        current_pos += len(audio_segment)
    
    return np.concatenate(result_audio) if result_audio else np.array([])


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    i = 0
    while i < len(sentences):
        if len(sentences[i].split()) < 4:
            if i == 0:
                # Merge with the next sentence
                sentences[i + 1] = sentences[i] + ', ' + sentences[i + 1]
                del sentences[i]
            else:
                # Merge with the previous sentence
                sentences[i - 1] = sentences[i - 1] + ', ' + sentences[i]
                del sentences[i]
                i -= 1
        else:
            i += 1

    final_sentences = []
    for sentence in sentences:
        parts = [p.strip() for p in sentence.split(', ')]
        buffer = []
        for part in parts:
            buffer.append(part)
            total_words = sum(len(p.split()) for p in buffer)
            if total_words > 20:
                # Split into separate chunks
                long_part = ', '.join(buffer)
                final_sentences.append(long_part)
                buffer = []
        if buffer:
            final_sentences.append(', '.join(buffer))

    if len(final_sentences[-1].split()) < 4 and len(final_sentences) >= 2:
        final_sentences[-2] = final_sentences[-2] + ", " + final_sentences[-1]
        final_sentences = final_sentences[0:-1]

    return final_sentences


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 6
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
    mel_spec_type=mel_spec_type,
    vocab_file="",
    ode_method=ode_method,
    use_ema=True,
    device=device,
):
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer)
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, clip_short=True, show_info=print, device=device):
    show_info("Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 15s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                        show_info("Audio is over 15s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            # 3. if no proper silence found for clipping
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                show_info("Audio is over 15s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    # Compute a hash of the reference audio file
    with open(ref_audio, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    if not ref_text.strip():
        global _ref_audio_cache
        if audio_hash in _ref_audio_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_audio_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_audio_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]


def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    # Validate silence tokens first
    is_valid, error_msg = validate_silence_tokens(gen_text)
    if not is_valid:
        show_info(f"Warning: {error_msg}")
        show_info("Proceeding without silence token processing...")
    else:
        # Count silence tokens for info
        silence_count = len(re.findall(r'<<<sil#(\d+)>>>', gen_text))
        if silence_count > 0:
            show_info(f"Found {silence_count} silence tokens")
    
    # Split the input text into batches (keeping silence tokens for now)
    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    for i, gen_text_batch in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text_batch)
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr),
            ref_text,
            gen_text_batches,
            model_obj,
            vocoder,
            mel_spec_type=mel_spec_type,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
    )


# infer batches


def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        local_speed = speed
        if len(gen_text.encode("utf-8")) < 10:
            local_speed = 0.3

        # Parse silence tokens from this batch
        cleaned_gen_text, silence_segments = parse_silence_tokens(gen_text)
        
        # Prepare the text
        text_list = [ref_text + cleaned_gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration based on cleaned text
            ref_text_len = len(ref_text.encode("utf-8"))
            gen_text_len = len(cleaned_gen_text.encode("utf-8"))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)

        # inference
        with torch.inference_mode():
            generated, _ = model_obj.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )
            del _

            generated = generated.to(torch.float32)  # generated mel spectrogram
            generated = generated[:, ref_audio_len:, :]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos":
                generated_wave = vocoder.decode(generated)
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()
            
            # Apply silence segments to this batch if any
            if silence_segments:
                generated_wave = apply_silence_to_audio(
                    generated_wave,
                    silence_segments,
                    sample_rate=target_sample_rate,
                    text_length=len(cleaned_gen_text)
                )

            if streaming:
                for j in range(0, len(generated_wave), chunk_size):
                    yield generated_wave[j : j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy()
                del generated
                yield generated_wave, generated_cpu

    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
            for future in progress.tqdm(futures) if progress is not None else futures:
                result = future.result()
                if result:
                    generated_wave, generated_mel_spec = next(result)
                    generated_waves.append(generated_wave)
                    spectrograms.append(generated_mel_spec)

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()

def replace_silence_with_placeholders(text):
    """
    Thay thế tất cả silent token <<<sil#...>>> bằng placeholder __SIL0__, __SIL1__, ...
    Trả về text mới, list silent token, list placeholder
    """
    pattern = r'<<<sil#\d+>>>'
    silences = re.findall(pattern, text)
    placeholders = [f"__SIL{i}__" for i in range(len(silences))]
    text_with_ph = text
    for sil, ph in zip(silences, placeholders):
        text_with_ph = text_with_ph.replace(sil, ph, 1)
    return text_with_ph, silences, placeholders

def restore_silence_from_placeholders(text, silences, placeholders):
    """
    Thay thế các placeholder __SIL0__, __SIL1__, ... về lại silent token gốc
    """
    for ph, sil in zip(placeholders, silences):
        text = text.replace(ph, sil)
    return text
