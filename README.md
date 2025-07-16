---
title: F5 TTS Vietnamese 100h Demo
emoji: 💻
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference




---
tags:
  - text-to-speech
  - vietnamese
  - ai-model
  - deep-learning
license: cc-by-nc-sa-4.0
library_name: pytorch
datasets:
  - ViVoice
  - VLSP2021
  - VLSP2022
  - VLSP2023
model_name: F5-TTS-Vietnamese-1000h
language: vi
---

# 🛑 Important Note ⚠️  
This model is only intended for **research purposes**.  
**Access requests must be made using an institutional, academic, or corporate email**. Requests from public email providers will be denied. We appreciate your understanding.  

# 🎙️ F5-TTS-Vietnamese-1000h  
A compact fine-tuned version of F5-TTS trained on 1000 hours of Vietnamese speech.  

🔗 For more fine-tuning experiments, visit: https://github.com/nguyenthienhy/F5-TTS-Vietnamese.  

📜 **License:** [CC-BY-NC-SA-4.0](https://spdx.org/licenses/CC-BY-NC-SA-4.0) — Non-commercial research use only.  

---

## 📌 Model Details  
- **Dataset:** Vi-Voice, VLSP 2021, VLSP 2022, VLSP 2023
- **Total dataset durations:** 1000 hours
- **Data processing Technique:**
  - Remove all music background from audios, using facebook demucs model: https://github.com/facebookresearch/demucs
  - Do not use audio files shorter than 1 second or longer than 30 seconds.
  - Using Chunk-Large-Former Speech2Text model by Zalo-AI to filter audio which has bad transcript
  - Keep the default punctuation marks unchanged.
  - Normalize to lowercase format.
- **Training Configuration:**  
  - **Base Model:** F5-TTS_Base  
  - **GPU:** RTX 3090  
  - **Batch Size:** 3200 frames - 1.5 months for training    

---

## 📝 Usage  
To load and use the model, follow the example below:  

```bash
git clone https://github.com/nguyenthienhy/F5-TTS-Vietnamese
cd F5-TTS-Vietnamese
python -m pip install -e.
mv F5-TTS-Vietnamese-ViVoice/config.json F5-TTS-Vietnamese-ViVoice/vocab.txt
f5-tts_infer-cli \
--model "F5TTS_Base" \
--ref_audio ref.wav \
--ref_text "cả hai bên hãy cố gắng hiểu cho nhau" \
--gen_text "mình muốn ra nước ngoài để tiếp xúc nhiều công ty lớn, sau đó mang những gì học được về việt nam giúp xây dựng các công trình tốt hơn" \
--speed 1.0 \
--vocoder_name vocos \
--vocab_file F5-TTS-Vietnamese-ViVoice/vocab.txt \
--ckpt_file F5-TTS-Vietnamese-ViVoice/model_last.pt \