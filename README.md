## Trợ Lý AI Đa Phương Thức: Dịch Thuật & Tóm Tắt (Anh - Việt)
### Hướng dẫn tải Mô hình (Model Weights)

Do giới hạn về dung lượng lưu trữ của GitHub, các tệp trọng số (weights) của hệ thống Trợ lý AI Đa phương thức đã được lưu trữ trên nền tảng đám mây. Để chạy được dự án, vui lòng tải các mô hình dưới đây và đặt vào đúng thư mục cấu hình:

### 1. Mô hình Qwen (Đã Merged)
Mô hình ngôn ngữ lớn Qwen đã được tinh chỉnh và merge trọng số để tối ưu hóa khả năng tương tác.
* **Link tải:** [Tải Qwen model](https://drive.google.com/drive/folders/1cal71LcVOnK-REktlsPCmWFpvGhz6Piy?usp=sharing)(#)
* **Vị trí lưu:** Giải nén và đặt toàn bộ tệp vào thư mục `models/qwen_merged/`

### 2. Mô hình BLIP (Image Understanding)
Mô hình xử lý thị giác máy tính dùng để đọc hiểu và phân tích hình ảnh đầu vào.
* **Link tải:** [Tải BLIP Model](https://drive.google.com/drive/folders/10Uz8upBCkQQilm2ueEnglJm_xc13KFYs?usp=sharing)(#)
* **Vị trí lưu:** Đặt tệp `.pth` hoặc `.bin` vào thư mục `models/blip/`

### 3. Cài đặt công cụ Tesseract OCR (Trích xuất văn bản)
Hệ thống sử dụng phần mềm **Tesseract OCR** để thực hiện chức năng bóc tách và đọc hiểu văn bản từ hình ảnh. Để hệ thống hoạt động trên môi trường Windows, vui lòng thực hiện các bước sau:

* **Bước 1:** Tải file cài đặt (Windows Installer) mới nhất tại đây: [Tải Tesseract OCR (64-bit)](https://github.com/UB-Mannheim/tesseract/wiki)
* 
* **Bước 2:** Tiến hành cài đặt phần mềm vào máy tính. 
  *(Lưu ý: Trong quá trình cài đặt, ở mục **Additional language data**, hãy nhớ tick chọn tải thêm gói ngôn ngữ `Vietnamese` nếu bạn muốn AI nhận diện văn bản tiếng Việt).*
  
* **Bước 3 (Quan trọng):** Cấu hình đường dẫn trong source code. Đảm bảo bạn đã trỏ đúng đường dẫn thực thi của Tesseract trong file xử lý Python. 
  
  *Ví dụ đoạn code cấu hình:*
  ```python
  import pytesseract
  # Sửa lại đường dẫn này cho khớp với thư mục bạn đã cài đặt trên máy
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
---

### 📂 Cấu trúc thư mục yêu cầu sau khi tải
Đảm bảo bạn đã sắp xếp các tệp mô hình đúng theo cấu trúc dưới đây trước khi chạy ứng dụng:

```text
📦 Your project name
 ┣ 📂 models
 ┃ ┣ 📂 qwen_merged
 ┃ ┃ ┣ 📜 config.json
 ┃ ┃ ┗ 📜 model.safetensors 
 ┃ ┣ 📂 blip
 ┃ ┃ ┗ 📜 blip_weights.pth
 ┃ ┗ 📂 ocr
 ┃   ┗ 📜 best_ocr_model.pth
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

### Giới thiệu dự án
- Dự án này là một hệ thống Trợ lý AI đa phương thức, được xây dựng dựa trên việc tinh chỉnh (Fine-tuning) mô hình ngôn ngữ lớn Qwen2.5-3B-Instruct. Hệ thống không chỉ có khả năng dịch văn bản song ngữ Anh - Việt với độ chính xác và ngữ cảnh tự nhiên cao, mà còn tích hợp khả năng tóm tắt văn bản sau khi dịch.
- Đặc biệt, dự án kết hợp khả năng xử lý đa phương thức (Multimodal). Người dùng có thể cung cấp dữ liệu đầu vào là hình ảnh; hệ thống sẽ tự động trích xuất văn bản (bằng công cụ OCR) hoặc sinh ra câu mô tả hình ảnh đó (bằng mô hình BLIP) để tiếp tục thực hiện luồng dịch thuật và tóm tắt.

### Tính năng nổi bật
- Dịch thuật Song ngữ: Dịch chính xác văn bản từ tiếng Anh sang tiếng Việt và ngược lại.
- Tóm tắt Thông minh: Tự động tóm tắt ngắn gọn nội dung văn bản sau khi dịch (khoảng 3 câu).
- Xử lý Đa phương thức (Multimodal):Tích hợp Tesseract OCR để nhận diện và bóc tách chữ viết từ hình ảnh tải lên. Tích hợp mô hình BLIP để tự động mô tả ngữ cảnh hình ảnh trong trường hợp không có hoặc có quá ít văn bản.
- Giao diện Chatbot Trực quan: Ứng dụng Web Chatbot thân thiện được phát triển bằng Flask, hỗ trợ giao tiếp và tương tác mượt mà.

### Công nghệ & Mô hình sử dụng
- Mô hình cốt lõi: Qwen/Qwen2.5-3B-Instruct.
- Kỹ thuật Fine-tuning: LoRA (Low-Rank Adaptation) với tham số tối ưu (r=16, alpha=32) giúp giảm thiểu tài nguyên huấn luyện.
- Định dạng dữ liệu: Instruction Tuning chuẩn ChatML.
- Computer Vision: Tesseract OCR (bóc tách chữ) , BLIP (Salesforce/blip-image-captioning-base) (mô tả ảnh).
- Backend: Python 3.10+, Flask.
- Thư viện AI: transformers, peft, trl, torch

###  Dữ liệu Huấn luyện (Dataset)
- Nguồn dữ liệu: Kho lưu trữ mã nguồn mở Hugging Face (harouzle/vi_en-translation).
- Quy mô: 100.000 mẫu cặp câu song ngữ (được tăng cường 2 chiều từ 50.000 mẫu gốc).
- Tiền xử lý: Chuẩn hóa mã Unicode NFKC, làm sạch khoảng trắng dư thừa, định dạng prompt hướng dẫn.

### Hướng dẫn Cài đặt & Triển khai
1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.10+. Clone repository và cài đặt các thư viện cần thiết:
```
Bash
git clone https://github.com/TranNhuKhaY512/EnVi-Qwen.git
cd EnVi-Qwen
pip install flask transformers peft torch pytesseract pillow
```
2. Cài đặt Tesseract OCR
- Hệ thống yêu cầu cài đặt phần mềm Tesseract OCR trên máy tính để đọc ảnh.
- Tải và cài đặt Tesseract tại: UB-Mannheim/tesseract (Nhớ chọn gói ngôn ngữ Vietnamese).
- Cấu hình: Cập nhật lại đường dẫn Tesseract trong mã nguồn Backend (file app.py) cho khớp với máy của bạn:
```
Python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
3. Tải Mô hình (Weights)
- Tạo thư mục models/ và tải các trọng số (weights) đã được huấn luyện vào đúng cấu trúc:
- Mô hình Qwen (LoRA Merged): [Tải Qwen Model](https://drive.google.com/drive/folders/1cal71LcVOnK-REktlsPCmWFpvGhz6Piy?usp=sharing) -> Giải nén vào models/qwen-merged/.
- Mô hình BLIP: Sẽ tự động tải qua Hugging Face trong lần chạy đầu tiên, hoặc bạn có thể tải tay và đặt vào models/blip/.

4. Khởi động hệ thống
Sau khi hoàn tất cấu hình, khởi động Web Server:
```
Bash
python app.py
```
- Truy cập ứng dụng qua trình duyệt tại địa chỉ: http://127.0.0.1:5000.

---

## 📞 Thông tin liên hệ

Nếu bạn có bất kỳ câu hỏi nào về dự án hoặc muốn trao đổi thêm, vui lòng liên hệ với đại diện nhóm:

* **Trần Như Khả Ý** 
* **Email:** trannhukhayy0512@gmail.com 
* **Điện thoại:** 0364551205
* **GitHub:** [TranNhuKhaY512](https://github.com/TranNhuKhaY512) 
* **LinkedIn:** [linkedin.com/in/trannhukhay051205](https://linkedin.com/in/trannhukhay051205) 
