# Ổn rồi hahaaha
import spaces
import os
import gradio as gr
import tempfile

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
    save_spectrogram,
    chunk_text,
    process_text_chunks,
)

def post_process(text):
    """
    Post-process text while preserving silence syntax
    """
    # First, temporarily replace silence tokens with placeholders
    import re
    silence_pattern = r'\s*<<<\s*sil#(\d{1,5})\s*>>>\s*'
    silence_tokens = []
    
    def replace_silence(match):
        ms = int(match.group(1))
        # Round to nearest 100ms and clamp to valid range
        ms = max(100, min(20000, int(round(ms / 100.0) * 100)))
        placeholder = f"__SILENCE_TOKEN_{len(silence_tokens)}__"
        silence_tokens.append(f" <<<sil#{ms}>>> ")
        return placeholder
    
    # Replace silence tokens with placeholders
    text_with_placeholders = re.sub(silence_pattern, replace_silence, text)
    
    # Apply normal post-processing to the text without silence tokens
    text_processed = " " + text_with_placeholders + " "
    text_processed = text_processed.replace(" . . ", " . ")
    text_processed = " " + text_processed + " "
    text_processed = text_processed.replace(" .. ", " . ")
    text_processed = " " + text_processed + " "
    text_processed = text_processed.replace(" , , ", " , ")
    text_processed = " " + text_processed + " "
    text_processed = text_processed.replace(" ,, ", " , ")
    text_processed = " " + text_processed + " "
    text_processed = text_processed.replace('"', "")
    text_processed = " ".join(text_processed.split())
    
    # Restore silence tokens
    for i, silence_token in enumerate(silence_tokens):
        placeholder = f"__SILENCE_TOKEN_{i}__"
        text_processed = text_processed.replace(placeholder, silence_token)
    
    return text_processed

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
def infer_tts(ref_audio_orig: str, gen_text: str, ref_text: str = "", speed: float = 1.0, request: gr.Request = None):
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    if len(gen_text.split()) > 10000:
        raise gr.Error("Please enter text content with less than 10000 words.")

    try:
        print(f"Original text: {gen_text}")
        print(f"Reference text: {ref_text}")
        
        # Post-process the text
        processed_text = post_process(gen_text)
        print(f"Post-processed text: {processed_text}")
        
        # Chunk the text
        processed_text_chunks = chunk_text(processed_text)
        print(f"Chunked text: {processed_text_chunks}")
        
        # Process chunks (apply TTSnorm to non-silence chunks)
        final_chunks = process_text_chunks(processed_text_chunks)
        print(f"Final chunks: {final_chunks}")
        
        ref_audio, ref_text_final = preprocess_ref_audio_text(ref_audio_orig, ref_text)
        
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, ref_text_final.lower(), final_chunks, model, vocoder, speed=speed
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
    # 🎤 F5-TTS: Tổng hợp giọng nói tiếng Việt từ văn bản.
    # Mô hình được huấn luyện với khoảng 1000 giờ dữ liệu trên GPU RTX 3090.
    Nhập văn bản và tải lên một mẫu giọng nói để tạo ra giọng nói tự nhiên.
    """)

    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(label="📝 Text", placeholder="Enter the text to generate voice...", lines=3)
    
    ref_text = gr.Textbox(label="📝 Reference Text (optional)", placeholder="If provided, will be used as reference text instead of ASR.", lines=2)

    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Speed")
    btn_synthesize = gr.Button("🔥 Generate Voice")

    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    model_limitations = gr.Textbox(
        value="""1. Mô hình này có thể không hoạt động tốt với các ký tự số, ngày tháng, ký tự đặc biệt, v.v. => Cần có một mô-đun chuẩn hóa văn bản. ( Đã cập nhật với vinorm TTSnorm)
    2. Nhịp điệu của một số đoạn âm thanh được tạo ra có thể không nhất quán hoặc bị giật => Nên chọn các đoạn âm thanh mẫu phát âm rõ ràng và ít ngắt quãng để có chất lượng tổng hợp tốt hơn.
    3. Mặc định, văn bản âm thanh tham chiếu sử dụng mô hình pho-whisper-medium, mô hình này có thể không nhận diện chính xác tiếng Việt, dẫn đến chất lượng tổng hợp giọng nói kém. ( Đã cập nhật với phần điền tay cho văn bản tham chiếu)
    4. Suy luận với các đoạn văn quá dài có thể cho kết quả kém. ( Đã cập nhật với thực thi trên Colab, khỏe hơn nên đã tăng giới hạn lên 10000 từ)""", 
        label="❗ Giới hạn của Mô hình",
        lines=4,
        interactive=False
    )


    btn_synthesize.click(infer_tts, inputs=[ref_audio, gen_text, ref_text, speed], outputs=[output_audio, output_spectrogram])

# Run Gradio with share=True to get a gradio.live link
demo.queue().launch(share=True)