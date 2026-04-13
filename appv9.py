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

# Load Qwen Merged Model (merge_19875)
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
    text = re.sub(r'(?<=\D)0(?=\D)', 'o', text)
    text = re.sub(r'(?<=\D)1(?=\D)', 'l', text)
    text = text.replace("|", "I")

    # 🔥 FIX LỖI KIỂU "aTag", "bTag", "xTag"
    text = re.sub(r'\b[a-zA-Z]{1,3}Tag\b', '', text)

    # 🔥 Xóa từ vô nghĩa (1–2 ký tự không phải số)
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
def detect_task(text):
    text = text.lower()

    has_summary = any(k in text for k in ["tóm tắt", "tom tat", "summarize", "summary"])
    has_translation = any(k in text for k in ["dịch", "translate"])

    if has_summary and has_translation:
        return "both"
    elif has_summary:
        return "summary"
    elif has_translation:
        return "translation"
    else:
        return "chat"

# --- 3. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/chat-process', methods=['POST'])
def chat_process():
    user_text = request.form.get('text', '').strip()
    image = request.files.get('image')

    # xử lý input
    img_text = ""
    if image:
        img_text = extract_content_from_image(image)

    if image and user_text:
        final_input = f"{user_text}\n\n[IMAGE_CONTENT]\n{img_text}"
    elif image:
        final_input = f"Mô tả nội dung sau:\n{img_text}"
    else:
        final_input = user_text

    # 🔥 PHẢI CÓ DÒNG NÀY
    task = detect_task(final_input)
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
    
    
    content = final_input.strip()

# 🔥 CLEAN mạnh hơn
    content = content.strip().strip('"').strip("'")
    lang = detect_language(content)

# ✅ SET LANGUAGE THEO TASK (QUAN TRỌNG)
    if task == "summary":
        lang_instruction = "Giữ nguyên ngôn ngữ gốc. TUYỆT ĐỐI KHÔNG dịch."

    elif task == "translation":
        lang_instruction = "Dịch sang ngôn ngữ còn lại (Việt ↔ Anh)."

    elif task == "both":
        lang_instruction = "Tóm tắt giữ nguyên ngôn ngữ, sau đó dịch sang ngôn ngữ còn lại."

    else:  # chat
        if lang == "vi":
            lang_instruction = "Trả lời bằng tiếng Việt."
        else:
            lang_instruction = "Answer in English."

    if task == "summary":
        instruction = f"""Summarize the following text in 2–3 sentences.

CRITICAL RULES:
- KEEP the original language
- DO NOT translate under any condition
- DO NOT add explanations
- ONLY output the summary

Text:
{content}
"""

    elif task == "translation":
        instruction = f"""Translate the following text to the other language (Vietnamese ↔ English).

CRITICAL RULES:
- ONLY translate
- DO NOT summarize
- DO NOT explain
- DO NOT add content

Text:
{content}
"""

    elif task == "both":
        instruction = f"""Perform EXACTLY 2 steps:

Step 1: Summarize the text in 2–3 sentences (keep original language)
Step 2: Translate the SUMMARY into the other language

CRITICAL RULES:
- Translation MUST be based ONLY on the summary
- DO NOT use the original text for translation
- DO NOT skip steps

OUTPUT FORMAT:

=== SUMMARY ===
<summary here>

=== TRANSLATION ===
<translation here>

Text:
{content}
"""

    else:  # chat
        instruction = content
    final_input = final_input.replace("<|user|>", "").replace("<|assistant|>", "")
# ===========================
    # Tạo Prompt theo định dạng của Qwen
    # Bạn có thể điều chỉnh System Prompt để chuyên biệt hóa dịch/tóm tắt
    # Trong file app.py, phần xây dựng prompt:

# --- SYSTEM PROMPT ĐA NĂNG ---
    system_instruction = """
You are a multi-functional AI with ONLY 4 tasks:
1. Translation (Vietnamese ↔ English)
2. Summarization
3. Summarization + Translation
4. Chat

CORE RULES:
- Perform ONLY ONE task based on user request
- NEVER combine tasks unless explicitly required
- Be concise, accurate, and follow instructions strictly
- User instruction has highest priority

CRITICAL CONSTRAINTS:

1. TRANSLATION:
- Translate to the other language only (VI ↔ EN)
- DO NOT summarize
- DO NOT explain
- DO NOT add content

2. SUMMARY:
- Summarize in 2–3 sentences
- Keep original language
- DO NOT translate

3. BOTH:
Step 1: Summarize (original language)
Step 2: Translate the SUMMARY

FORMAT:

=== SUMMARY ===
...

=== TRANSLATION ===
...

4. CHAT:
- Natural, short, helpful

IMPORTANT:
- NEVER change task
- NEVER mix tasks
- ALWAYS follow output language instruction
"""
# Xây dựng Prompt cuối cùng gửi tới Model
    prompt = f"""<|system|>
{system_instruction}

<|user|>
{instruction}

Yêu cầu ngôn ngữ: {lang_instruction}

<|assistant|>
"""
   # --- CẤU HÌNH STREAMING ---
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# ✅ DÁN Ở ĐÂY
    if task == "summary":
            generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=256,
        temperature=0,
        do_sample=False,
        top_p=1,
        repetition_penalty=1.05
    )

    elif task == "translation":
        generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.05
    )

    elif task == "both":
        generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=600,
        temperature=0.15,
        do_sample=True,
        top_p=0.85,
        repetition_penalty=1.05
    )

    else:
        generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        temperature=0.6,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.05
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
    