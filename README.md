**👉 [Google Colab / F5-TTS-VN-hynt.ipynb](https://colab.research.google.com/drive/1PgW8jEEAmuTxaKKe49cyallVQZ0N9jho?usp=sharing)**



---

## 🎯 Tính năng chính

| Tính năng | Mô tả |
|-----------|-------|
| **Text-to-Speech** | Chuyển đổi văn bản tiếng Việt thành giọng nói tự nhiên |
| **Voice Cloning** | Nhân bản giọng nói từ mẫu âm thanh reference |
| **Tốc độ linh hoạt** | Tùy chỉnh tốc độ phát âm theo nhu cầu |
| **Tạo khoảng lặng** | Tùy chỉnh khoảng im lặng bất kì trong văn bản đầu vào, tăng chân thật |

---

## ⚙️ Yêu cầu hệ thống

### 📊 Phần cứng
- **GPU**: vì dùng CPU rất chậm và không ổn định (T4 trở lên trên Google Colab)

### 📁 Model files
#### Đảm bảo có đủ các file sau: `model_last.pt`, `config.json`

#### **Nguồn tải về:**
  - 📂 [Google Drive](https://drive.google.com/drive/folders/1JSQUKc74IxF4Fng9zg5RA17AE-1RtNWT?usp=drive_link)
  - 🤗 [Hugging Face](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice)

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Chuẩn bị môi trường
```bash
# Chuyển Runtime sang GPU trong Google Colab
# Thời gian khởi động: 2-3 phút
```

### Bước 2: Chuẩn bị model
Chọn một trong hai phương pháp:

#### 🔗 Phương pháp 1: Mount Google Drive *(Khuyên dùng)*
1. Thêm thư mục [Models](https://drive.google.com/drive/folders/1JSQUKc74IxF4Fng9zg5RA17AE-1RtNWT?usp=drive_link) vào "My Drive"
2. Chạy mount Google Drive

#### 📥 Phương pháp 2: Clone model
- Tải trực tiếp từ repository
- *Lưu ý: Chậm hơn và kém ổn định*

### Bước 3: Chọn nguồn model
- **Google Drive**: Cần cấu hình đường dẫn
- **Hugging Face**: Tự động, không cần cấu hình

### Bước 4: Khởi động ứng dụng
```bash
# Thời gian khởi động lần đầu: 2-3 phút
# Chờ đến khi xuất hiện URL: https://xxxxxxxxxxx.gradio.live
```

---

## 🎛️ Giao diện người dùng

### Các thành phần chính

| Thành phần | Mô tả | Ghi chú |
|------------|-------|---------|
| **Sample Voice** | Upload file âm thanh `ref_audio` | 6-15 giây, chất lượng cao, không tạp âm |
| **Text** | Nhập `gen_text` cần chuyển đổi | sửa `app.py` bỏ/tăng giới hạn 10000 words  |
| **Reference Text** | Nội dung của `ref_audio` | Nếu `ref_audio` rõ ràng thì nên bỏ trống, auto transcribe |
| **Generate Voice** | Nút bắt đầu chuyển đổi | Kết quả hiển thị phía dưới |

---

## 🔇 CÚ PHÁP TẠO KHOẢNG IM LẶNG

> 🎯 **TÍNH NĂNG ĐẶC BIỆT**: F5-TTS Vietnamese hỗ trợ tạo khoảng im lặng có chủ đích trong văn bản!

---

### 🎵 Định dạng chuẩn

```markdown
<<<sil#{number_milisecond}>>>
```

| Thông số | Mô tả | Phạm vi |
|----------|-------|---------|
| **number_milisecond** | Thời gian im lặng (mili giây) | 100 - 20,000 ms |
| **Làm tròn** | Tự động làm tròn số | 110→100, 150→200, 9990→10000 |

---

### 🎯 Ví dụ sử dụng

#### ✅ **CÁC CÁCH DÙNG ĐÚNG**

```markdown
# Khoảng lặng 1 giây
Xin chào <<<sil#1000>>> các bạn!

# Khoảng lặng 2 giây
Câu đầu tiên. <<<sil#2000>>> Câu thứ hai.

# Khoảng lặng 500ms
Đây là <<<sil#500>>> một ví dụ ngắn.

# Khoảng lặng trong văn bản dài
Chương một <<<sil#1500>>> nói về lịch sử. <<<sil#1000>>> Chương hai <<<sil#2000>>> nói về tương lai.
```

#### ❌ **CÁC CÁCH DÙNG SAI**

```markdown
# SAI: Thiếu khoảng trắng trước
Xin chào<<<sil#1000>>> các bạn!

# SAI: Thiếu khoảng trắng sau  
Xin chào <<<sil#1000>>>các bạn!

# SAI: Thiếu cả hai khoảng trắng
Xin chào<<<sil#1000>>>các bạn!

# SAI: Chèn giữa từ (gây phát âm sai)
tuoi <<<sil#1000>>> tre.com  # Đọc: "Tuổi tê rờ e chấm cơm"
```

---


### 🚨 **QUY TẮC QUAN TRỌNG**

| ⚠️ Quy tắc | Mô tả | Ví dụ |
|------------|-------|-------|
| **Khoảng trắng bắt buộc** | Phải có space trước và sau | `text <<<sil#1000>>> text` |
| **Không tách từ** | Không chèn giữa từ/cụm từ | ❌ `VN <<<sil#1000>>> ESE` |
| **Số dương** | Chỉ dùng số nguyên dương | 100-20,000 |
| **Cú pháp chính xác** | Đúng format với dấu < > # | `<<<sil#1000>>>` |

---

### 💡 **CÁC TRƯỜNG HỢP LỖI THƯỜNG GẶP**

#### 🔴 Lỗi cú pháp - Kết quả: *"Bé hơn bé hơn bé hơn ét i lờ..."*

```markdown
<<<sil#1000>>>-          # Có ký tự không phải space
<<<sil#-1000>>>         # Số âm
<<<sil#>>>              # Thiếu số
<<<silnce#1000>>>       # Sai chính tả
<<<si#1000>>>           # Thiếu chữ 'l'
<<sil#1000>>            # Thiếu dấu <
<<<sil@1000>>>          # Sai ký tự @
```

---

### 🎯 **TIPS SỬ DỤNG HIỆU QUẢ**

#### 📝 Các tình huống thực tế:

```markdown
# Tạo nhịp cho bài thơ
Mùa xuân đến rồi <<<sil#1000>>> 
Hoa nở khắp nơi <<<sil#1500>>>
Chim ca líu lo <<<sil#1000>>>
Lòng ta vui thơ <<<sil#2000>>>

# Tạo khoảng lặng trong bài thuyết trình
Xin chào mọi người! <<<sil#1000>>> Hôm nay tôi sẽ trình bày về <<<sil#500>>> công nghệ AI.

# Tạo hiệu ứng kịch tính
Và kết quả là <<<sil#2000>>> thành công hoàn toàn!
```

---

### 📊 **BẢNG THỜI GIAN THAM KHẢO**

| Thời gian | Ứng dụng | Ví dụ |
|-----------|----------|-------|
| **100-300ms** | Ngắt nhẹ | Dấu phẩy <<<sil#200>>> ngắt câu |
| **500-1000ms** | Ngắt vừa | Kết thúc câu <<<sil#800>>> bắt đầu câu mới |
| **1000-2000ms** | Ngắt rõ ràng | Chuyển chủ đề <<<sil#1500>>> nội dung mới |
| **2000ms+** | Ngắt dài | Hiệu ứng kịch tính <<<sil#3000>>> |

---

> 💡 **MẸO**: Sử dụng khoảng im lặng giúp văn bản nghe tự nhiên hơn, tạo nhịp điệu và dễ theo dõi!

---

## 🔧 Xử lý sự cố

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|-------------|-----------|
| **Lỗi OOM** | Thiếu bộ nhớ | Giảm batch size hoặc độ dài text |
| **Model không load** | Đường dẫn sai | Kiểm tra đường dẫn file model |
| **Chất lượng âm thanh kém** | File reference kém | Sử dụng file âm thanh chất lượng cao |

---

## 📊 Hiệu suất

> **Thống kê thực tế**: Với văn bản Lão Hạc (~16,000 ký tự) trên T4 GPU Google Colab:
> - ⏱️ Thời gian xử lý: 20 phút
> - 🔊 Đầu ra: Audio dài 20 phút

---

## 🌐 Nguồn tham khảo

### Repositories
- 📂 [F5-TTS-Vietnamese](https://github.com/nguyenthienhy/F5-TTS-Vietnamese)
- 🤗 [Hugging Face Space](https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h)

### Models
- 🤗 [F5-TTS-Vietnamese-ViVoice](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice)

---

## 🚧 Tính năng đang phát triển

- **NGROK API Server**: Đang cập nhật...

---











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

source: 
- https://github.com/nguyenthienhy/F5-TTS-Vietnamese
- https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h
- https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice
