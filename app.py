import spaces
import os
import gradio as gr
import tempfile
from vinorm import TTSnorm
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
    save_spectrogram,
    parse_silence_tokens,
    apply_silence_to_audio,
    replace_silence_with_placeholders,
    restore_silence_from_placeholders,
    chunk_text,
)
from f5_tts.model import DiT

# Retrieve token from secrets

def post_process(text):
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = " " + text + " "
    text = text.replace(" .. ", " . ")
    text = " " + text + " "
    text = text.replace(" , , ", " , ")
    text = " " + text + " "
    text = text.replace(" ,, ", " , ")
    text = " " + text + " "
    text = text.replace('"', "")
    return " ".join(text.split())

def calculate_relative_positions(original_text, processed_text, silence_segments):
    """
    Calculate relative positions of silence tokens after text processing
    
    Args:
        original_text (str): Original text before processing
        processed_text (str): Text after TTSnorm processing
        silence_segments (list): List of (position, duration) tuples from original text
        
    Returns:
        list: Updated silence segments with adjusted positions
    """
    if not silence_segments:
        return []
    
    # Simple proportional mapping based on text length ratio
    original_len = len(original_text)
    processed_len = len(processed_text)
    
    if original_len == 0:
        return []
    
    ratio = processed_len / original_len
    adjusted_segments = []
    
    for pos, duration in silence_segments:
        # Calculate new position proportionally
        new_pos = int(pos * ratio)
        # Ensure position is within bounds
        new_pos = min(new_pos, processed_len)
        adjusted_segments.append((new_pos, duration))
    
    return adjusted_segments

# Load models from local checkpoint
model_dir = os.path.join(os.path.dirname(__file__), "F5-TTS-Vietnamese-ViVoice")
ckpt_file = os.path.join(model_dir, "model_last.pt")
vocab_file = os.path.join(model_dir, "config.json")

vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=ckpt_file,
    vocab_file=vocab_file,
)

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, request: gr.Request = None):
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    if len(gen_text.split()) > 1000:
        raise gr.Error("Please enter text content with less than 1000 words.")
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        # 1. Thay silent token bằng placeholder
        text_with_ph, silences, placeholders = replace_silence_with_placeholders(gen_text)
        # 2. Chuẩn hóa text (vinorm, post_process) trên text_with_ph
        processed_text = post_process(TTSnorm(text_with_ph)).lower()
        # 3. Chia batch trên processed_text (vẫn còn placeholder)
        batches = chunk_text(processed_text)
        # 4. Với mỗi batch, khôi phục lại silent token từ placeholder
        batches_with_silence = [restore_silence_from_placeholders(batch, silences, placeholders) for batch in batches]
        # 5. Gộp lại thành text cuối cùng để truyền vào infer_process
        final_text = ' '.join(batches_with_silence)
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, ref_text.lower(), final_text, model, vocoder, speed=speed
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)
        return (final_sample_rate, final_wave), spectrogram_path
    except Exception as e:
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis.
    # The model was trained with approximately 1000 hours of data on an RTX 3090 GPU. 
    Enter text and upload a sample voice to generate natural speech.
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(label="📝 Text", placeholder="Enter the text to generate voice...", lines=3)
    
    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Speed")
    btn_synthesize = gr.Button("🔥 Generate Voice")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    model_limitations = gr.Textbox(
        value="""1. This model may not perform well with numerical characters, dates, special characters, etc. => A text normalization module is needed.
2. The rhythm of some generated audios may be inconsistent or choppy => It is recommended to select clearly pronounced sample audios with minimal pauses for better synthesis quality.
3. Default, reference audio text uses the pho-whisper-medium model, which may not always accurately recognize Vietnamese, resulting in poor voice synthesis quality.
4. Inference with overly long paragraphs may produce poor results.""", 
        label="❗ Model Limitations",
        lines=4,
        interactive=False
    )

    btn_synthesize.click(infer_tts, inputs=[ref_audio, gen_text, speed], outputs=[output_audio, output_spectrogram])

# Run Gradio with share=True to get a gradio.live link
demo.queue().launch(share=True)