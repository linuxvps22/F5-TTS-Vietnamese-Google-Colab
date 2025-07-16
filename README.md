**👉 [Google Colab / F5-TTS-VN-hynt.ipynb](https://colab.research.google.com/drive/1PgW8jEEAmuTxaKKe49cyallVQZ0N9jho?usp=sharing)**



## 📖 Hướng dẫn sử dụng
### 🎯 Các tính năng chính:
- **Text-to-Speech**: Chuyển đổi text tiếng Việt thành giọng nói tự nhiên
- **Voice Cloning**: Nhân bản giọng nói từ mẫu âm thanh
- **Điều chỉnh tốc độ**: Tùy chỉnh tốc độ phát âm

### 💡 Lưu ý quan trọng:
1. **GPU**: Dùng GPU, trong Colab thì có T4 đổ lên
2. **Model files**: Đảm bảo đã có file `model_last.pt` và `config.json`, có thể tải từ ([Drive](https://drive.google.com/drive/folders/1JSQUKc74IxF4Fng9zg5RA17AE-1RtNWT?usp=drive_link) hoặc [Huggingface](https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice))
3. **Thời gian khởi động**: Lần đầu chạy sẽ mất 2-3 phút để load model
4. **Chất lượng âm thanh**: Sử dụng file âm thanh reference chất lượng cao, không tạp âm, giới hạn **6 ≤ 15** giây

### 🔧 Troubleshooting:
- **Lỗi OOM**: Giảm batch size hoặc độ dài text
- **Model không load**: Kiểm tra đường dẫn file model
- **Chất lượng âm thanh kém**: Sử dụng file reference tốt hơn

---

## Nguồn
- **Original Repository**: [F5-TTS-Vietnamese](https://github.com/nguyenthienhy/F5-TTS-Vietnamese)
- **Hugging Face Space**: [hynt/F5-TTS-Vietnamese-100h](https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h)


## Thao tác

1. Chuyển Runtime thành sử dụng GPU và chạy bước 1 ( 2 - 3 phút)
2. Chuẩn bị model ở bước 2 với 1 trong 2 phương pháp:
    - Thêm lối tắt thư mục [Models](https://drive.google.com/drive/folders/1JSQUKc74IxF4Fng9zg5RA17AE-1RtNWT?usp=drive_link) được chia sẻ vào phần "Drive của tôi" / "My drive", để luôn ở đó hoặc là đường dẫn tùy chỉnh. Sau đó chạy phương pháp 1 **Mount Google Drive** ( Nhanh gọn lẹ)
    - Chạy phương pháp 2 **Clone model** ( Lâu, kém ổn định)

3. Ở bước 3, chọn nguồn lấy models từ Drive hoặc Huggingface, nếu nguồn là Huggingface thì không cần cấu hình đường dẫn nữa.

4. Chạy bước 4 để khởi động chương trình, lần đầu chạy sẽ mất khoảng **2 - 3** phút. Chờ đến khi xuất hiện thấy **url** dạng **https://xxxxxxxxxxx.gradio.live**, đó là giao diện, click vào **url** đó để mở Gradio GUI

5. Các thành phần trên giao diện:
  - **Sample Voice** để upload giọng tham chiếu `ref_audio` ( giọng gốc để clone, nên rõ ràng, không tạp âm, giới hạn **6 ≤ 15** giây) 
  - **Text** để nhập văn bản `gen_text` ( văn bản sẽ được chuyển thành giọng nói chạy trên Colab thì sửa nội dung file app.py bỏ hoặc tăng giới hạn, trên Huggingface thì nên giữ 1000 và thử với text ngắn)
  - **Reference Text (optional)** để nhập nội dung của file audio **Sample Voice**, nên bỏ trống nếu **Sample Voice** rõ ràng, chỉ nên chỉnh sửa khi kiểm tra log trong Colab sau khi chạy **Generate Voice** và thấy log hiển thị `ref_text` bị sai hoặc không hợp lý. Nếu không nhập sẽ tự chuyển thành text ( transcribe)
  - **Generate Voice** nút thực hiện bắt đầu chuyển văn bản thành giọng nói, sau khi hoàn thành sẽ hiển thị kết quả ngay phía bên dưới. Trong quá trình, nếu có lỗi xảy ra sẽ được log tại cell bước 4 trong Colab


6. Cú pháp tạo khoảng im lặng có chủ đích với input text:

    ## Định dạng Cú pháp Silence Utterance
    ```
    <<<sil#{number_milisecond}>>>
    ```

    ## Tham số
    - **number_milisecond**: Một số nguyên dương trong khoảng từ 100 đến 20000 để xác định thời lượng im lặng, tính bằng mili giây.
    - **number_milisecond** sẽ được làm tròn lên. Ví dụ: 110 sẽ thành 100, 150 sẽ thành 200, 10001 sẽ thành 10000, 9990 sẽ thành 10000 và tương tự.

    ## Quy tắc Quan trọng

    ### Yêu cầu về Khoảng trắng
    Trước và sau cú pháp silence cần phải có ký tự **khoảng trắng**, nếu không sẽ không hoạt động.

    #### ✅ Ví dụ Hoạt động:
    ```
    This is <<<sil#1000>>> 1 second silence
    ```

    #### ❌ Ví dụ KHÔNG Hoạt động:
    ```
    This is<<<sil#1000>>> 1 second silence
    ```

    ```
    This is <<<sil#1000>>>1 second silence
    ```

    ```
    This is<<<sil#1000>>>1 second silence
    ```

    ### Cảnh báo về Chèn vào Từ/Cụm từ
    Silence utterance **không nên** được chèn vào giữa một từ hoặc cụm từ, nếu không sẽ tạo ra những phát âm không mong muốn.

    #### Ví dụ:
    - **Bình thường**: tuoitre.com
      - *đọc* "tuổi trẻ chấm cơm"

    - **Sai**: tuoi <<<sil#1000>>> tre.com
      - *đọc* "Tuổi tê rờ e chấm cơm"

    ## Các Trường hợp Lỗi

    ### Ví dụ Cú pháp Sai
    Trong trường hợp cú pháp sai, việc tạo silence utterances sẽ trở nên khó nghe:

    1. **Có ký tự không phải khoảng trắng ở trước hoặc sau**: `<<<sil#1000>>>-`
      - Cụm từ trên sẽ được đọc bằng tiếng Việt như thế này *"Bé hơn bé hơn bé hơn ét i lờ thăng một không không không lớn hơn lớn hơn lớn hơn"*

    2. **{number_milisecond} là số âm**: `<<<sil#-1000>>>`
      - *đọc* "Bé hơn bé hơn bé hơn ét i lờ thăng một không không không lớn hơn lớn hơn lớn hơn"

    3. **{number_milisecond} để trống**: `<<<sil#>>>`
      - *đọc* "Bé hơn bé hơn bé hơn ét i lờ thăng lớn hơn lớn hơn lớn hơn"

    4. **Các ví dụ cú pháp sai khác**:
      - `abc<<<sil#1000>>>`
      - `<<<silnce#1000>>>`
      - `<<<si#1000>>>`
      - `<<sil#1000>>`
      - `<<<sil@1000>>>`

    ## Tóm tắt
    - Luôn sử dụng khoảng trắng đúng cách xung quanh cú pháp silence
    - Sử dụng số nguyên dương từ 100-20000 cho mili giây
    - Không tách từ hoặc cụm từ bằng cú pháp silence
    - Tuân theo định dạng chính xác: `<<<sil#{number}>>>`

7. NGROK API server ( Updating... )

***Fact: Với text là một bài văn Lão Hạc hơn 16000 ký tự thì với T4 GPU của Google Colab chạy hết 20 phút và cho ra output dài 20 phút.***

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
