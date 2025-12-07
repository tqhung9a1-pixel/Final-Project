import streamlit as st
from convert import image_to_base64
from convert import font_to_base64
from main import run_prediction
import base64

# CẤU HÌNH TRANG
st.set_page_config(
    page_title="Đồ án cuối kì - Nhóm 3",
    page_icon="🎬",
    layout="centered"
)

# FONT & LOGO
font_base64 = font_to_base64("font.ttf")
logo_base64 = image_to_base64("logo.png")

# CSS THIẾT KẾ GIAO DIỆN
st.markdown(f"""
    <style>
    @font-face {{
        font-family: 'Montserrat';
        src: url(data:font/ttf;base64,{font_base64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}

    /* MỞ RỘNG KHU VỰC NỘI DUNG */
    .block-container {{
        max-width: 1200px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    body {{
        background: #121212;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(40,40,40,0.5) 0%, transparent 20%),
            radial-gradient(circle at 80% 70%, rgba(40,40,40,0.5) 0%, transparent 20%);
        background-attachment: fixed;
        background-size: cover;
    }}

    /* LOGO TRÊN GÓC TRÁI */
    .logo {{
        position: fixed;
        top: 15px;
        left: 15px;
        width: 12vw;
        max-width: 165px;
        min-width: 90px;
        z-index: 2147483647;
    }}

    /* CONTAINER GIỚI THIỆU */
    .apple-container-dark {{
        background: #0b0b0b;
        padding: 35px;
        border-radius: 28px;
        overflow: hidden;
        margin: 40px auto;
        max-width: 900px;
        box-shadow:
            0 0 30px rgba(255,255,255,0.06),
            0 0 60px rgba(255,255,255,0.04);
    }}

    .intro-inside {{
        color: #dddddd;
        font-size: 20px;
        line-height: 1.6;
        text-align: center;
        font-family: 'Montserrat', sans-serif;
    }}

    /* STYLE NÚT – KHÔNG DÙNG POSITION FIXED */
    div.stButton > button {{
        background: #1a1a1a;
        color: #ffffff;
        border: 2px solid #cccccc;
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: bold;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.4);
        transition: all 0.3s ease;
        width: 100% !important;
        display: block;
        margin: 0 auto;
    }}

    div.stButton > button:hover {{
        background: #ffffff;
        color: #000000;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.8);
        transform: scale(1.05);
    }}

    /* KHUNG UPLOAD */
    .upload-box {{
        background: #1e1e1e;
        border: 2px dashed #444;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 20px auto;
        max-width: 600px;
        transition: all 0.3s ease;
    }}
    .upload-box:hover {{
        border-color: #00ffaa;
        box-shadow: 0 0 15px rgba(0,255,170,0.3);
    }}

    /* ẢNH ĐẦU VÀO */
    img.full-img {{
        width: 80% !important;
        height: auto !important;
        border-radius: 10px;
        margin: 20px auto;
        display: block;
    }}

    /* TIÊU ĐỀ KẾT QUẢ */
    .selected-title {{
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        margin: 10px 0;
        color: white;
        text-shadow: 0px 0px 6px black;
    }}

    /* TIÊU ĐỀ CHÍNH (GLOW) */
    .title-container {{
        text-align: center;
        width: 100%;
        margin: 30px 0;
    }}
    .title-wrapper {{
        position: relative;
        display: inline-block;
    }}
    .title-glow {{
        font-size: 77px;
        font-weight: 900;
        font-family: 'Montserrat', sans-serif;
        color: transparent;
        text-shadow:
            0 0 4px rgba(255,255,255,0.3),
            0 0 8px rgba(255,255,255,0.25),
            0 0 16px rgba(255,255,255,0.15);
        position: absolute;
        top: 0;
        left: 0;
        pointer-events: none;
        z-index: 1;
    }}
    .title-real {{
        font-size: 77px;
        font-weight: 900;
        font-family: 'Montserrat', sans-serif;
        color: #1e1e1e;
        text-shadow:
            0 0 2px rgba(255,255,255,0.85),
            0 0 4px rgba(255,255,255,0.65);
        position: relative;
        z-index: 2;
    }}
    </style>

    <img src="data:image/png;base64,{logo_base64}" class="logo">
""", unsafe_allow_html=True)

# TIÊU ĐỀ
st.markdown("""
<div class="title-container">
    <div class="title-wrapper">
        <div class="title-glow">ĐỒ ÁN CUỐI KÌ - NHÓM 3</div>
        <div class="title-real">ĐỒ ÁN CUỐI KÌ - NHÓM 3</div>
    </div>
</div>
""", unsafe_allow_html=True)

# GIỚI THIỆU
st.markdown("""
<div class="apple-container-dark">
    <div class="intro-inside">
        Bạn có tin liệu một model Machine Learning cơ bản có thể phân biệt ngày và đêm thông qua những bức ảnh?<br><br>
        Hãy thử thách model của chúng mình bằng chính bức ảnh của bạn ☀️🌙
    </div>
</div>
""", unsafe_allow_html=True)

# HƯỚNG DẪN UPLOAD
st.markdown('<div style="text-align: center; font-size: 18px; margin: 20px 0;">Hãy bỏ vào bức ảnh bạn muốn phân tích 📷 :</div>', unsafe_allow_html=True)

# KHUNG UPLOAD (tự căn giữa do block-container đã set max-width)
uploaded_file = st.file_uploader(
    "", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# NÚT "PHÂN TÍCH" – CĂN GIỮA BẰNG CỘT
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyze_clicked = st.button(
        " Phân tích ", key="analyze_btn", use_container_width=True)

# XỬ LÝ KHI NHẤN NÚT
if analyze_clicked:
    if uploaded_file is None:
        st.warning("Vui lòng tải lên một bức ảnh trước khi phân tích!")
    else:
        img_bytes = uploaded_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()

        st.markdown(
            '<div class="selected-title">Ảnh bạn đã chọn</div>', unsafe_allow_html=True)
        st.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="full-img">', unsafe_allow_html=True)

        # Lưu tạm
        save_path = "anh-cua_minh.jpg"
        with open(save_path, "wb") as f:
            f.write(img_bytes)

        # Gọi model
        with st.spinner("🔍 Đang phân tích hình ảnh, vui lòng chờ..."):
            result_label, fig1, fig2 = run_prediction(image_path=save_path)

        # HIỂN THỊ KẾT QUẢ
        st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
            <h3 style="color: #00ff88;">Phân tích thành công!</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="text-align: center; font-size: 28px; color: white; font-weight: bold; margin: 10px 0;">
                🌞🌙 {result_label}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Biểu đồ 1
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            st.pyplot(fig1, use_container_width=True)

        # Biểu đồ 2
        st.markdown(
            '<div class="selected-title">Ảnh vẽ lại bằng 5 màu nổi bật</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            st.pyplot(fig2, use_container_width=True)
