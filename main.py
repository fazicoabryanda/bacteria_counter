import streamlit as st
import base64
from PIL import Image
import os

# ---- PAGE CONFIG (must be first) ----
st.set_page_config(
    page_title="Colony Counter App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Optional import check ----
ultra_import_error = None
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    ultra_import_error = e

# ---- Paths ----
MODEL_WEIGHTS = "ecoli.pt"
BACKGROUND_VECTOR = "vector1.png"
ILLUSTRATION_IMAGE = "ilustrasi_utama.png"

# ---- Utils & Global CSS ----
@st.cache_data(show_spinner=False)
def get_base64_of_bin_file(bin_file: str) -> str:
    try:
        with open(bin_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

def set_png_as_page_bg(png_file: str):
    bin_str = get_base64_of_bin_file(png_file)
    bg_style = ""
    if bin_str:
        bg_style = f"""
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: right top;
        background-repeat: no-repeat;
        """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #FEF9FF !important;
            {bg_style}
            min-height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def apply_global_css():
    st.markdown(
        """
        <style>
        /* --------- Typography (Public Sans) --------- */
        @import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Public Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
            color: #1F1F39;
        }

        /* DO NOT HIDE header; make it clean & transparent so the sidebar toggle stays visible */
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 56px; 
        }
        /* Ensure the collapsed sidebar chevron is always visible */
        div[data-testid="collapsedControl"] {
            display: block !important;
            color: #A057D0 !important;
            z-index: 9999 !important;
        }

        /* You can still hide the footer if you like */
        footer { visibility: hidden; height: 0; }

        /* Headings */
        h1 { font-size: 3.2rem !important; line-height: 1.1 !important; font-weight: 700 !important; }
        h2 { color:#1F1F39 !important; border-left: 6px solid #A35AEE; padding-left: 12px; font-weight: 700 !important; }
        h5 { font-size: 1.1rem !important; color: #4A5568 !important; }
        h6 { font-size: 0.95rem !important; font-weight: 600 !important; color: #2C3E50 !important; }

        /* Tabs */
        div[data-testid="stTabs"] button p { color:#1F1F39 !important; font-weight:600; }
        div[data-testid="stTabs"] button[aria-selected="true"] { border-bottom-color:#A35AEE !important; }

        /* Image card wrapper */
        .image-card { background:transparent; border:none; border-radius:0; box-shadow:none; padding:0; margin-bottom:20px; text-align:center; }
        .image-card h4 { color:#1F1F39; font-weight:600; margin-bottom:10px; }

        /* File uploader spacing */
        div[data-testid="stFileUploader"] > section { padding-top: 0; }

        /* Success box */
        .stSuccess {
            background:#1C4D2E !important; color:#fff !important;
            border-left:6px solid #28A745 !important; border-radius:10px;
            padding:14px 16px; margin-bottom:20px;
        }

        /* --------- SIDEBAR (purple) --------- */
        aside[aria-label="sidebar"] { background: #A057D0 !important; color: #FFFFFF !important; }
        aside[aria-label="sidebar"] * { color: #FFFFFF !important; }
        aside[aria-label="sidebar"] h1, 
        aside[aria-label="sidebar"] h2, 
        aside[aria-label="sidebar"] h3 { letter-spacing:.2px; font-weight:700 !important; }

        /* Vertical, tidy buttons */
        aside[aria-label="sidebar"] .stButton > button {
            width: 100% !important;
            display: block !important;
            padding: 12px 14px !important;
            margin: 6px 0 12px 0 !important;
            border-radius: 12px !important;
            background: rgba(255,255,255,0.16) !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            text-align: left;
            box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
        }
        aside[aria-label="sidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.24) !important;
            transform: translateY(-1px);
        }

        /* Sidebar expander */
        aside[aria-label="sidebar"] details > summary { font-weight:700 !important; }
        aside[aria-label="sidebar"] details {
            background: rgba(255,255,255,0.10) !important;
            border-radius: 12px !important;
            padding: 4px 6px;
        }

        /* Sidebar sliders & checkboxes */
        aside[aria-label="sidebar"] .stSlider > div > div,
        aside[aria-label="sidebar"] label { color:#FFFFFF !important; }
        aside[aria-label="sidebar"] .stSlider [role="slider"] { border: 2px solid #FFFFFF !important; }
        aside[aria-label="sidebar"] .stSlider [data-baseweb="slider"] > div > div { background: rgba(255,255,255,0.35) !important; }
        aside[aria-label="sidebar"] .stCheckbox label { color:#FFFFFF !important; font-weight:600 !important; }

        /* Small caption color */
        aside[aria-label="sidebar"] .stCaption { color:#F2E9FF !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---- Model helpers ----
@st.cache_resource
def load_yolo_model(model_path: str):
    if YOLO is None:
        return None
    if not os.path.exists(model_path):
        return f"Error: Model file '{model_path}' not found."
    try:
        return YOLO(model_path)
    except Exception as e:
        return f"Failed to load YOLO: {e}"

def run_prediction(model, image: Image.Image):
    if (model is None) or isinstance(model, str):
        st.warning("Model is not loaded. Check `ecoli.pt` or install `ultralytics`.")
        st.session_state.prediction_ran = True
        return

    conf_val = st.session_state.conf_slider_global / 100
    iou_val = st.session_state.iou_slider_global
    agnostic_val = st.session_state.agnostic_checkbox_global

    with st.spinner("Analyzing colonies..."):
        try:
            results = model(
                image,
                conf=conf_val,
                iou=iou_val,
                agnostic_nms=agnostic_val,
                verbose=False,
            )
            if results and len(results) > 0:
                result = results[0]
                arr = result.plot(labels=True, masks=True, boxes=True)  # BGR
                st.session_state.predicted_image = Image.fromarray(arr[..., ::-1])  # to RGB
                st.session_state.colony_count = len(result.boxes)
            else:
                st.session_state.predicted_image = None
                st.session_state.colony_count = 0

            st.session_state.prediction_ran = True
            st.rerun()
        except Exception as e:
            st.error(f"Failed to run prediction: {e}")
            st.session_state.prediction_ran = True
            st.rerun()

# ---- Pages ----
def home_page():
    set_png_as_page_bg(BACKGROUND_VECTOR)

    left, right = st.columns([6, 4], gap="large")
    with left:
        st.markdown("<h1>Count Bacteria Colonies<br>In Seconds</h1>", unsafe_allow_html=True)
        st.markdown(
            "<h5>Transform Petri dish images into reliable colony counts with modern computer vision. "
            "Standardize results across operators and sessions—no more manual bias, no more tedious counting.</h5>",
            unsafe_allow_html=True,
        )
        if st.button("Get Started", key="start_app_btn", type="primary"):
            st.session_state.page = "Colony Counter"
            st.rerun()

        st.markdown("<h6>Why It Fits Your Lab</h6>", unsafe_allow_html=True)
        st.markdown(
            """
            <ul>
                <li>Intuitive & Accessible</li>
                <li>Consistent, Reproducible Results</li>
                <li>Collaboration-Ready Outputs</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if os.path.exists(ILLUSTRATION_IMAGE):
            st.image(ILLUSTRATION_IMAGE, output_format="PNG")
        else:
            st.markdown("<br><br><br><br>", unsafe_allow_html=True)
            st.info(f"Placeholder: please add `{ILLUSTRATION_IMAGE}`.")

def counter_page(model):
    set_png_as_page_bg(BACKGROUND_VECTOR)

    st.markdown("<h2>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Bacteria Colonies Counter</h2>", unsafe_allow_html=True)
    st.caption("Upload a Petri dish image or use the camera. Prediction runs automatically. Tune detection in the **NMS Settings** on the sidebar.")

    if st.session_state.uploaded_image is not None:
        with st.container():
            st.markdown("---")
            col_ori, col_pred = st.columns(2, gap="large")

            def image_card(column, title, image, prediction_ran):
                with column:
                    st.markdown("<div class='image-card'>", unsafe_allow_html=True)
                    st.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
                    if image is not None:
                        st.image(image, width=360)
                    else:
                        if not prediction_ran:
                            st.image(st.session_state.uploaded_image, width=360, caption="Waiting for prediction…")
                        else:
                            st.warning("No result returned.")
                    st.markdown("</div>", unsafe_allow_html=True)

            image_card(col_ori, "Original Image", st.session_state.uploaded_image, st.session_state.prediction_ran)
            image_card(col_pred, "Predicted Image", st.session_state.predicted_image, st.session_state.prediction_ran)
            st.markdown("---")

    if st.session_state.prediction_ran and st.session_state.uploaded_image is not None and st.session_state.colony_count is not None:
        if st.session_state.colony_count > 0:
            st.markdown(
                f"""
                <div class='stSuccess'>
                    <h3>✅ Colonies counted: {st.session_state.colony_count}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No colonies were detected.")
    elif st.session_state.prediction_ran and st.session_state.uploaded_image is not None:
        st.error("Could not get the colony count.")

    in_col, _ = st.columns([2, 1])
    with in_col:
        st.markdown("<h6>Choose Input Method</h6>", unsafe_allow_html=True)
        tab_upload, tab_camera = st.tabs(["🖼️ File Upload", "📸 Camera"])

        with tab_upload:
            st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                key="upload_file_key",
                label_visibility="hidden",
            )
            uploaded_file = st.session_state.upload_file_key
            if uploaded_file is not None:
                try:
                    current_img = Image.open(uploaded_file).convert("RGB")
                    current_img.name = uploaded_file.name
                    if st.session_state.uploaded_image is None or current_img.name != getattr(st.session_state.uploaded_image, "name", "N/A"):
                        st.session_state.uploaded_image = current_img
                        st.session_state.predicted_image = None
                        st.session_state.colony_count = None
                        st.session_state.prediction_ran = False
                        run_prediction(model, st.session_state.uploaded_image)
                except Exception as e:
                    st.error(f"Failed to process file: {e}")

        with tab_camera:
            camera_image = st.camera_input("Camera", key="camera_input_key", label_visibility="hidden")
            if camera_image is not None:
                try:
                    current_img = Image.open(camera_image).convert("RGB")
                    current_img.name = "Camera Capture"
                    if (
                        st.session_state.uploaded_image is None
                        or getattr(st.session_state.uploaded_image, "name", "N/A") != "Camera Capture"
                        or st.session_state.predicted_image is not None
                    ):
                        st.session_state.uploaded_image = current_img
                        st.session_state.predicted_image = None
                        st.session_state.colony_count = None
                        st.session_state.prediction_ran = False
                        run_prediction(model, st.session_state.uploaded_image)
                except Exception as e:
                    st.error(f"Failed to process camera input: {e}")

# ---- App ----
def main():
    apply_global_css()

    if ultra_import_error is not None:
        st.error("`ultralytics` is not installed. Please run: `pip install ultralytics`")

    # Session state defaults
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("uploaded_image", None)
    st.session_state.setdefault("predicted_image", None)
    st.session_state.setdefault("colony_count", None)
    st.session_state.setdefault("prediction_ran", False)
    st.session_state.setdefault("conf_slider_global", 40.0)
    st.session_state.setdefault("iou_slider_global", 0.5)
    st.session_state.setdefault("agnostic_checkbox_global", False)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔬 BIO-LAB COUNTER")
        st.caption("Version 1.0")

        # Vertical nav buttons
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()

        if st.button("🦠 Counter", use_container_width=True):
            st.session_state.page = "Colony Counter"
            st.rerun()

        st.markdown("---")

        def on_nms_change():
            if st.session_state.uploaded_image is not None:
                st.session_state.prediction_ran = False
                st.session_state.predicted_image = None

        with st.expander("🛠️ NMS Settings", expanded=False):
            st.slider(
                "Confidence Threshold (%)",
                0.0, 100.0, st.session_state.conf_slider_global, 5.0,
                key="conf_slider_global", on_change=on_nms_change,
                help="Minimum confidence required to keep a detection.",
            )
            st.slider(
                "IoU Threshold",
                0.01, 0.99, st.session_state.iou_slider_global, 0.05,
                key="iou_slider_global", on_change=on_nms_change,
                help="Intersection over Union threshold for NMS.",
            )
            st.checkbox(
                "Agnostic NMS",
                st.session_state.agnostic_checkbox_global,
                key="agnostic_checkbox_global", on_change=on_nms_change,
                help="Apply NMS without using class labels.",
            )

        st.markdown("---")
        st.caption("Powered by YOLOv8 Instance Segmentation.")

    # Router
    if st.session_state.page == "Home":
        home_page()
    else:
        yolo_model = load_yolo_model(MODEL_WEIGHTS)
        counter_page(yolo_model)

if __name__ == "__main__":
    main()

