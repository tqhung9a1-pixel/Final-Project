import streamlit as st
from convert import image_to_base64
from convert import font_to_base64
from main1 import run_prediction
import base64

# CẤU HÌNH TRANG
st.set_page_config(
    page_title="Đồ án cuối kì - Nhóm 3",
    page_icon="🎬",
    layout="centered"
)
# FONT
font_base64 = font_to_base64("font.ttf")
# LOGO
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
    /*MỞ RỘNG TOÀN BỘ GIAO DIỆN*/
    .block-container {{
        max-width: 90% !important;
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
    .apple-container-dark {{
        background: #0b0b0b;
        padding: 35px;
        border-radius: 28px;
        position: relative;
        overflow: hidden;
        margin: 40px auto;
        max-width: 900px;
        box-shadow:
            0 0 30px rgba(255,255,255,0.06),
            0 0 60px rgba(255,255,255,0.04);
    }}

    .apple-container-dark::before {{
        content: "";
        position: absolute;
        top: -30%;
        left: -30%;
        width: 160%;
        height: 160%;
        background: radial-gradient(
            circle,
            rgba(255,255,255,0.10) 0%,
            rgba(255,255,255,0.04) 40%,
            rgba(255,255,255,0.00) 75%
        );
        filter: blur(45px);
        z-index: 0;
    }}

    .intro-inside {{
        position: relative;
        z-index: 2;
        color: #dddddd;
        font-size: 20px;
        line-height: 1.6;
        text-align: center;
        font-family: 'Montserrat', sans-serif;
    }}

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
    }}

    div.stButton > button:hover {{
        background: #ffffff;
        color: #000000;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.8);
        transform: scale(1.05);
    }}
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
    .logo {{
    position: fixed;
    top: 15px;
    left: 15px;
    width: 500px;
    z-index: 2147483647;
    }}
    </style>
    <img src="data:image/png;base64,{logo_base64}" class="logo">
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* ÉP ẢNH FULL WIDTH */
img.full-img {
    width: 80% !important;
    height: auto !important;
    border-radius: 10px;
    margin: auto;
    display: block;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.selected-title {
    text-align: center;
    font-size: 24px;          /* chữ to hơn */
    font-weight: 700;         /* đậm */
    margin-top: -10px;        /* kéo gần ảnh hơn */
    margin-bottom: 10px;      /* cách ảnh 1 chút cho đẹp */
    color: white;             /* màu nổi hơn */
    text-shadow: 0px 0px 6px black; /* nhìn rõ trên nền tối */
}
</style>
""", unsafe_allow_html=True)

# TIÊU ĐỀ VÀ GIỚI THIỆU
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@900&display=swap');
.title-container {
    text-align: center;
    width: 100%;
    margin: 30px 0;
}
.title-wrapper {
    position: relative;
    display: inline-block;
}
/* Lớp glow - lan tỏa mềm, nhẹ */
.title-glow {
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
}
/* CHỮ CHÍNH: Tối hơn tí, viền sáng mềm tan */
.title-real {
    font-size: 77px;
    font-weight: 900;
    font-family: 'Montserrat', sans-serif;
    color: #1e1e1e !important;  /* Tối hơn so với #2c2c2c */
    text-shadow:
        0 0 2px rgba(255,255,255,0.85),
        0 0 4px rgba(255,255,255,0.65);
    position: relative;
    z-index: 2;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="title-container">
    <div class="title-wrapper">
        <div class="title-glow">ĐỒ ÁN CUỐI KÌ - NHÓM 3</div>
        <div class="title-real">ĐỒ ÁN CUỐI KÌ - NHÓM 3</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="apple-container-dark">
    <div class="intro-inside">
        Bạn có tin liệu một model Machine Learning cơ bản có thể phân biệt ngày và đêm thông qua những bức ảnh?
        <br><br>
        Hãy thử thách model của chúng mình bằng chính bức ảnh của bạn ☀️🌙
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="intro-text">Hãy bỏ vào bức ảnh bạn muốn phân tích 📷 :</div>',
            unsafe_allow_html=True)

# UPLOAD ẢNH VÀ NÚT PHÂN TÍCH
uploaded_file = st.file_uploader(
    "", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
with col5:
    analyze_clicked = st.button(" Phân tích ", key="analyze_btn")

if analyze_clicked:
    if uploaded_file is None:
        st.warning("Vui lòng tải lên một bức ảnh trước khi phân tích!")
    else:
        # Đọc ảnh ngay và hiển thị trước khi phân tích
        img_bytes = uploaded_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()

        st.markdown(
            "<div class='selected-title'>Ảnh bạn đã chọn</div>", unsafe_allow_html=True)

        st.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="full-img">',
            unsafe_allow_html=True
        )

        # Lưu file tạm
        save_path = "anh-cua_minh.jpg"
        with open(save_path, "wb") as f:
            f.write(img_bytes)

        # Hiệu ứng chờ
        with st.spinner("🔍 Đang phân tích hình ảnh, vui lòng chờ..."):
            result_label, fig1, fig2 = run_prediction(image_path=save_path)

        # Kết quả căn giữa
        st.markdown("""
        <div style="text-align: center;">
            <h3 style="color: #00ff88;">Phân tích thành công!</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="text-align: center; font-size: 28px; color: white; font-weight: bold; margin-top: 10px;">
                🌞🌙 {result_label}
            </div>
            """,
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns([0.5, 3, 0.5])
        with c2:
            st.pyplot(fig1, use_container_width=True)
        c1, c2, c3 = st.columns([0.5, 3, 0.5])
        with c2:
            st.markdown(
                "<div class='selected-title'>Ảnh vẽ lại bằng 5 màu nổi bật</div>", unsafe_allow_html=True)
            st.pyplot(fig2, use_container_width=True)
