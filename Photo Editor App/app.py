import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ==================================
# 1. Page Config & CSS
# ==================================
st.set_page_config(page_title="Pro Photo Editor", page_icon="🎨", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ==================================
# 2. Language & Text Management
# ==================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'English'

# زر تبديل اللغة
col_l1, col_l2 = st.columns([8, 2])
with col_l2:
    # استخدام radio أو selectbox حسب نسخة ستريم ليت
    lang_choice = st.radio("Language / اللغة", ["English", "العربية"], horizontal=True,
                           index=0 if st.session_state.lang == "English" else 1)
    st.session_state.lang = lang_choice

# قاموس النصوص (تم تعديله ليناسب تطبيق الصور)
texts = {
    "English": {
        "title": "🎨 Pro Photo Editor Studio",
        "subtitle": "Resize, Filter, and Enhance your images instantly.",
        "upload": "Upload Image",
        "sidebar_header": "⚙️ Control Panel",
        "resize_sec": "Resize & Aspect Ratio",
        "filter_sec": "Filters & Effects",
        "orig_img": "Original Image",
        "edit_img": "Edited Image",
        "download": "Download Image",
        "filters": ['Original', 'Black and White', 'Pencil Sketch', 'Brightness', 'HDR', 'Style']
    },
    "العربية": {
        "title": "🎨 استوديو تعديل الصور الاحترافي",
        "subtitle": "تغيير الحجم، فلاتر، وتحسين الصور بضغطة زر.",
        "upload": "ارفع الصورة هنا",
        "sidebar_header": "⚙️ لوحة التحكم",
        "resize_sec": "تغيير الحجم والأبعاد",
        "filter_sec": "الفلاتر والمؤثرات",
        "orig_img": "الصورة الأصلية",
        "edit_img": "الصورة المعدلة",
        "download": "تحميل الصورة",
        "filters": ['أصلي', 'أبيض وأسود', 'رسم رصاص', 'إضاءة', 'HDR', 'ستايل كرتوني']
    }
}
L = texts[st.session_state.lang]

# العنوان الرئيسي
st.title(L["title"])
st.caption(L["subtitle"])
st.divider()


# ==================================
# 3. Processing Functions
# ==================================

def apply_resize(img, ratio_mode):
    # img is a numpy array (Height, Width, Channels)
    h, w = img.shape[:2]

    if ratio_mode == 'Original':
        return img

    # تحديد الأبعاد الجديدة بناء على النسبة المختارة
    # سنقوم بتثبيت العرض وتغيير الطول، أو العكس، لتبسيط الكود
    # هنا سنقوم بعمل Center Crop أو تغيير الحجم (بسيط)
    # للأسهل سنقوم بتغيير الحجم Force Resize

    target_width = 800  # عرض ثابت للجودة

    if ratio_mode == '1:1':
        new_h = target_width
    elif ratio_mode == '16:9':
        new_h = int(target_width * 9 / 16)
    elif ratio_mode == '4:3':
        new_h = int(target_width * 3 / 4)
    elif ratio_mode == '9:16':
        target_width = 450  # نقلل العرض عشان الطول ما يضربش
        new_h = int(target_width * 16 / 9)
    else:
        return img  # Fallback

    resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
    return resized


def apply_filter(img, filter_name, params):
    # تحويل الصورة للتنسيق المناسب لـ OpenCV عند الحاجة

    if filter_name in ['Black and White', 'أبيض وأسود']:
        # تحويل من RGB (PIL) إلى Gray
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    elif filter_name in ['Pencil Sketch', 'رسم رصاص']:
        # Sketch بيحتاج صورة رمادية أحياناً أو ملونة
        # الـ PencilSketch في CV2 بيرجع صورتين (Gray, Color)
        gray, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return gray  # نرجع نسخة الرصاص الرمادية

    elif filter_name in ['HDR', 'HDR']:
        return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

    elif filter_name in ['Brightness', 'إضاءة']:
        # نستخدم beta للتحكم في الإضاءة
        value = params.get('brightness_val', 30)
        return cv2.convertScaleAbs(img, alpha=1, beta=value)

    elif filter_name in ['Style', 'ستايل كرتوني']:
        return cv2.stylization(img, sigma_s=60, sigma_r=0.6)

    return img


# ==================================
# 4. Sidebar & Controls
# ==================================

with st.sidebar:
    st.header(L["sidebar_header"])

    # --- Resize Controls ---
    st.subheader(L["resize_sec"])
    size_option = st.selectbox('Select Ratio', ['Original', '1:1', '16:9', '4:3', '9:16'])

    # --- Filter Controls ---
    st.subheader(L["filter_sec"])
    filter_option = st.radio('Select Filter', L["filters"])

    # Dynamic Sliders based on filter
    filter_params = {}
    if filter_option in ['Brightness', 'إضاءة']:
        filter_params['brightness_val'] = st.slider('Brightness Level', -100, 100, 30)

# ==================================
# 5. Main App Logic
# ==================================

uploaded_file = st.file_uploader(L["upload"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة بـ PIL ثم تحويلها لـ Numpy Array
    image_pil = Image.open(uploaded_file)
    original_image_np = np.array(image_pil)

    # 1. تطبيق تغيير الحجم
    resized_img = apply_resize(original_image_np, size_option)

    # 2. تطبيق الفلتر
    processed_img = apply_filter(resized_img, filter_option, filter_params)

    # ================= View =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(L["orig_img"])
        st.image(original_image_np, use_container_width=True)

    with col2:
        st.subheader(L["edit_img"])

        # تحويل الصورة للعرض (Streamlit بيعرض Grayscale غلط لو مخدش parameter)
        if len(processed_img.shape) == 2:  # Grayscale
            st.image(processed_img, use_container_width=True, channels='GRAY')
            # تحويلها لـ RGB عشان التحميل يشتغل صح
            save_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
        else:
            st.image(processed_img, use_container_width=True)
            save_img = processed_img

        # ================= Download Button =================
        # تحويل الصورة لـ Bytes للتحميل
        # OpenCV بيشتغل BGR لما نيجي نسيف، بس هنا احنا معانا RGB من الـ Processing
        # لازم نحولها BGR قبل الـ Encoding
        save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".png", save_img_bgr)

        if is_success:
            st.download_button(
                label=f"📥 {L['download']}",
                data=io.BytesIO(buffer),
                file_name="edited_image.png",
                mime="image/png"
            )

else:
    st.info("👋 " + (
        "Please upload an image to start." if st.session_state.lang == "English" else "من فضلك ارفع صورة للبدء."))