import streamlit as st
from convert import image_to_base64, font_to_base64
from main import run_prediction
import base64

# PAGE CONFIG
st.set_page_config(
    page_title="Final Project - Group 3",
    page_icon="🎬",
    layout="centered"
)

# FONT & LOGO
font_base64 = font_to_base64("font.ttf")
logo_base64 = image_to_base64("logo.png")

# APPLY CSS FROM FILE
with open("style.css", "r", encoding="utf-8") as f:
    css_content = f.read().replace("{{FONT_BASE64}}", font_base64)
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# LOGO
st.markdown(
    f'<img src="data:image/png;base64,{logo_base64}" class="app-logo">', unsafe_allow_html=True)

# MAIN TITLE
st.markdown("""
<div class="title-section">
    <div class="title-wrapper">
        <div class="title-main">ĐỒ ÁN CUỐI KÌ - NHÓM 3</div>
    </div>
</div>
""", unsafe_allow_html=True)

# INTRODUCTION
st.markdown("""
<div class="intro-container">
    <div class="intro-text">
        Bạn có tin một model Machine Learning cơ bản có thể phân biệt ngày và đêm chỉ từ hình ảnh?<br><br>
        Thử thách model của chúng tôi với bức ảnh của bạn! ☀️🌙
    </div>
</div>
""", unsafe_allow_html=True)

# FILE UPLOADER
st.markdown('<div style="text-align: center; font-size: 18px; margin: 20px 0;">Hãy tải lên bức ảnh bạn muốn phân tích 📷 :</div>', unsafe_allow_html=True)

_, c, _ = st.columns([1, 6, 1])
with c:
    uploaded_file = st.file_uploader(
        " ", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# ANALYZE BUTTON CENTERED
_, c, _ = st.columns([2, 1, 2])
with c:
    analyze_clicked = st.button(
        "Phân tích", key="analyze_btn", use_container_width=True)

# PROCESS BUTTON CLICK
if analyze_clicked:
    if uploaded_file is None:
        st.warning("Vui lòng tải lên một bức ảnh trước khi phân tích!")
    else:
        img_bytes = uploaded_file.read()
        img_base64 = base64.b64encode(img_bytes).decode()

        # SHOW SELECTED IMAGE
        st.markdown('<div class="result-title">Ảnh bạn đã chọn</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="preview-image">', unsafe_allow_html=True)

        # SAVE TEMP IMAGE
        save_path = "anh-cua_minh.jpg"
        with open(save_path, "wb") as f:
            f.write(img_bytes)

        with st.status("🔍 Đang phân tích ảnh, vui lòng chờ...", expanded=True) as status:
            status.write("📤 Tải ảnh vào mô hình")
            result_label, fig1, fig2 = run_prediction(image_path=save_path)
            status.write("📊 Tạo biểu đồ và vẽ lại ảnh")
            status.update(label="Hoàn tất!", state="complete")

        # DISPLAY RESULT
        st.markdown(
            '<div style="text-align: center; margin: 30px 0;"><h3 style="color: #00ff88;">Phân tích thành công!</h3></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="result-label">🌞🌙 {result_label}</div>',
            unsafe_allow_html=True
        )
        tab1, tab2 = st.tabs(["📈 Trích xuất đặc trưng", "🎨 Ảnh được vẽ lại"])

        with tab1:
            st.markdown(
                '<div class="result-title">Phân tích đặc trưng ảnh</div>',
                unsafe_allow_html=True
            )
            _, c, _ = st.columns([1, 6, 1])
            with c:
                st.pyplot(fig1, use_container_width=True)

        with tab2:
            st.markdown(
                '<div class="result-title">Hình ảnh được vẽ lại với 5 màu chủ đạo</div>',
                unsafe_allow_html=True
            )
            _, c, _ = st.columns([1, 6, 1])
            with c:
                st.pyplot(fig2, use_container_width=True)
