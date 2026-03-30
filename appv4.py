import os
import torch
import re
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from PIL import Image
import pytesseract
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    BlipProcessor, 
    BlipForConditionalGeneration,
    TextIteratorStreamer
)
from threading import Thread

app = Flask(__name__)

# --- CẤU HÌNH HỆ THỐNG ---
# Trỏ đúng đường dẫn Tesseract trên máy bạn
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
device = "cuda" if torch.cuda.is_available() else "cpu"
# --- 1. LOAD MODELS ---
print("--- Đang khởi tạo Models... ---")
# Load BLIP để mô tả ảnh (nếu OCR thất bại)
blip_processor = BlipProcessor.from_pretrained("./models/blip")
blip_model = BlipForConditionalGeneration.from_pretrained("./models/blip").to(device)

# Load Qwen Merged Model 
model_path = "qwen-merged" 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)
print("--- Hệ thống đã sẵn sàng! ---")

# --- 2. CÔNG CỤ XỬ LÝ ---
# =========================
# CLEAN OCR TEXT (🔥 QUAN TRỌNG)
# =========================
def clean_ocr_text(text):
    if not text:
        return ""
    # 1. Chuẩn hóa unicode
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    # 2. Fix lỗi OCR phổ biến (có kiểm soát)
    text = re.sub(r'\b0\b', 'o', text)
    text = re.sub(r'\b1\b', 'l', text)
    text = text.replace("|", "I")

    # FIX LỖI KIỂU "aTag", "bTag", "xTag"
    text = re.sub(r'\b[a-zA-Z]{1,3}Tag\b', '', text)

    # Xóa từ vô nghĩa (1–2 ký tự không phải số)
    # Chỉ xóa ký tự rác thật sự
    text = re.sub(r'\b[a-zA-Z]{1}\b(?![a-z])', ' ', text)  # chỉ xóa chữ đơn đứng riêng

    # 3. Giữ ký tự hợp lệ
    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩ"
        r"òóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ"
        r".,!?;:()\-\n]",
        " ",
        text
    )

    # 4. Fix chữ dính
    text = re.sub(r'([a-zà-ỹ])([A-Z])', r'\1 \2', text)

    # 5. Fix dấu câu
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)

    # 6. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Viết hoa đầu câu
    def capitalize_sentences(text):
        if not text:
            return text

        text = text[0].upper() + text[1:]

        text = re.sub(
            r'([.!?]\s*)([a-zà-ỹ])',
            lambda m: m.group(1) + m.group(2).upper(),
            text
        )

        return text

    text = capitalize_sentences(text)

    return text

def extract_content_from_image(image_file):
    """Trích xuất chữ (OCR) hoặc mô tả ảnh (BLIP)"""
    try:
        image_file.seek(0)  # 🔥 QUAN TRỌNG
        img = Image.open(image_file).convert('RGB')
        # Thử OCR trước
        text = pytesseract.image_to_string(img, lang="vie+eng").strip()
        text = clean_ocr_text(text)  
        if len(text) < 15:
            # Nếu chữ quá ít, dùng BLIP mô tả hình ảnh
            inputs = blip_processor(img, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs, max_new_tokens=50)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            return f"[Hình ảnh hiển thị: {caption}]"
        return text
    except Exception as e:
        return f"[Lỗi xử lý ảnh: {str(e)}]"

# --- 3. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

# Route chính xử lý hội thoại (Chat Process)
@app.route('/chat-process', methods=['POST'])
def chat_process():
    user_text = request.form.get('text', '').strip()
    image = request.files.get('image')
    
    # ===== XỬ LÝ INPUT =====
    final_input = user_text

    if image:
        img_text = extract_content_from_image(image)
        if user_text:
            final_input = f"Nội dung ảnh: {img_text}\n\nYêu cầu: {user_text}"
        else:
            final_input = f"Hãy mô tả hoặc xử lý nội dung này: {img_text}"

    # ===== DETECT NGÔN NGỮ =====
    def detect_language(text):
        text = text.lower().strip()

        vi_chars = "ăâđêôơưàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹđ"
        vi_words = [
        "toi","la","va","mot","co","khong","ban","minh",
        "dang","duoc","nay","kia","roi","chua","gi","day",
        "sao","the","nao","lam","di","noi","yeu","thuong",
        "hoc","sinh","vien","anh","em","chi","toi","nay"
    ]

        words = text.split()

        vi_score = 0

    # 1. có dấu → +2 điểm mỗi ký tự
        vi_score += sum(2 for c in text if c in vi_chars)

    # 2. từ tiếng Việt → +1 điểm
        vi_score += sum(1 for w in words if w in vi_words)

    # 3. nếu điểm đủ lớn → tiếng Việt
        if vi_score >= 2:
            return "vi"

        return "en"
    lang = detect_language(final_input)

    if lang == "vi":
        lang_instruction = "Trả lời bằng tiếng Việt."
    else:
        lang_instruction = "Answer in English."
# ===========================
    # Tạo Prompt theo định dạng của Qwen
    # Bạn có thể điều chỉnh System Prompt để chuyên biệt hóa dịch/tóm tắt
    # Trong file app.py, phần xây dựng prompt:

# --- SYSTEM PROMPT ĐA NĂNG ---
    system_instruction = """
Bạn là AI chuyên:
- Dịch Anh ⇄ Việt
- Tóm tắt văn bản

=====================
🎯 NGUYÊN TẮC:
- Hiểu đúng yêu cầu trước khi trả lời
- Chỉ thực hiện đúng nhiệm vụ (dịch / tóm tắt / chat)
- Không thêm thông tin ngoài yêu cầu
- Không trộn ngôn ngữ

=====================
📌 QUY TẮC NGÔN NGỮ (ÁP DỤNG CHUNG):
- Viết đúng chính tả tiếng Việt hoặc tiếng Anh
- Tự động sửa lỗi chính tả nhẹ (đặc biệt lỗi OCR)
- Viết hoa đúng tên riêng (người, địa danh, tổ chức)
- Nếu tên riêng sai hoặc viết thường → tự sửa lại
- Ví dụ: ha noi → Hà Nội, viet nam → Việt Nam

=====================
📌 NHIỆM VỤ:

1. DỊCH:
- Xác định ngôn ngữ nguồn
- Dịch sang ngôn ngữ còn lại (Anh ↔ Việt)
- Giữ nguyên ý nghĩa, tự nhiên
- Không giải thích, không thêm nội dung

2. TÓM TẮT:
- Tóm tắt ngắn gọn nhưng đủ ý chính
- Viết 2–3 câu, rõ ràng, mạch lạc
- Diễn đạt lại (không sao chép nguyên văn)
- Gộp ý trùng, bỏ chi tiết phụ
- BẮT BUỘC giữ lại kết luận/quan điểm nếu có (đặt ở câu cuối)

3. HỘI THOẠI:
- Trả lời ngắn gọn, đúng trọng tâm

=====================
🌐 NGÔN NGỮ OUTPUT:
- Dịch → dùng ngôn ngữ đích
- Tóm tắt → giữ nguyên ngôn ngữ gốc
- Chat → theo ngôn ngữ người dùng

=====================
📦 FORMAT:

- DỊCH:
=== TRANSLATION ===
<nội dung dịch>

- TÓM TẮT:
=== SUMMARY ===
<nội dung tóm tắt>

- CHAT:
→ Trả lời bình thường (không tiêu đề)

=====================
⚠️ QUY TẮC CUỐI:
- Ưu tiên NGẮN GỌN + CHÍNH XÁC
- Không viết dài hơn cần thiết
- Không giải thích nếu không được yêu cầu
"""

# Xây dựng Prompt cuối cùng gửi tới Model
    prompt = f"<|system|>\n{system_instruction}\n<|user|>\n{final_input}\n<|assistant|>\n"
    # --- CẤU HÌNH STREAMING ---
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1
    )

    # Chạy generation trong thread riêng để không chặn Flask
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def generate():
        for new_text in streamer:
            # Trả về từng cụm từ ngay khi model sinh ra
            yield new_text

    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/clear', methods=['POST'])
def clear_history():
    # Xử lý xóa session hoặc lịch sử nếu cần
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    # Chạy server
    app.run(debug=False, host='0.0.0.0', port=5000)