"""
X-Ray Insight - web UI for the COVID-19 chest X-ray classifier.

Runs on the Azure ML compute instance and sends images to the deployed
managed online endpoint (covid-xray-aditi).

Start it from the repository root (so the dataset and outputs/ folder are found):
    streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 \
        --server.enableCORS false --server.enableXsrfProtection false
Then open:  https://<compute-instance-name>-8501.<region>.instances.azureml.ms
"""
import base64
import json
import os
import time
from datetime import datetime

import requests
import streamlit as st

ENDPOINT_NAME = "covid-xray-aditi"
MODEL_NAME = "covid-xray-resnet50"
CLASSES = ["Covid", "Normal", "Viral Pneumonia"]
LOW_CONFIDENCE = 0.60
SAMPLE_DIR = "Covid19-dataset/test"

CLASS_STYLE = {
    "Covid":           {"color": "#C62828", "bg": "#FDECEA", "icon": "🦠",
                        "note": "Patterns consistent with COVID-19 pneumonia."},
    "Normal":          {"color": "#2E7D32", "bg": "#E8F5E9", "icon": "✅",
                        "note": "No signs of pneumonia detected."},
    "Viral Pneumonia": {"color": "#E65100", "bg": "#FFF3E0", "icon": "🫁",
                        "note": "Patterns consistent with (non-COVID) viral pneumonia."},
}

st.set_page_config(page_title="X-Ray Insight", page_icon="🩻", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1150px;}
.hero h1 {margin-bottom: 0; font-size: 2.1rem;}
.hero p {color: #5f6b7a; margin-top: .2rem;}
.result-card {border-radius: 14px; padding: 18px 22px; margin: 6px 0 14px;}
.result-label {font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; opacity: .75;}
.result-class {font-size: 1.9rem; font-weight: 700; margin: 2px 0;}
.result-conf {font-size: 1rem;}
.bar-row {margin: 8px 0;}
.bar-head {display: flex; justify-content: space-between; font-size: .92rem;}
.bar-track {background: #EDF0F4; border-radius: 6px; height: 12px; overflow: hidden;}
.bar-fill {height: 12px; border-radius: 6px;}
.metric-box {background: #F5F7FA; border-radius: 12px; padding: 12px 14px; text-align: center;}
.metric-box b {font-size: 1.35rem; display: block;}
.metric-box span {color: #5f6b7a; font-size: .8rem;}
.disclaimer {font-size: .8rem; color: #7a8594;}
div.stButton > button[kind="primary"] {background: #0B3D91; border-color: #0B3D91; color: #fff; font-weight: 600;}
div.stButton > button[kind="primary"]:hover {background: #0A3277; border-color: #0A3277;}
.stTabs [aria-selected="true"] {color: #0B3D91 !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color: #0B3D91 !important;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- Azure connection
@st.cache_resource(show_spinner="Connecting to Azure ML endpoint...")
def get_endpoint():
    """Return (scoring_uri, key, model_version) for the deployed endpoint."""
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    ep = ml_client.online_endpoints.get(ENDPOINT_NAME)
    key = ml_client.online_endpoints.get_keys(ENDPOINT_NAME).primary_key
    try:
        version = ml_client.online_deployments.get("blue", ENDPOINT_NAME).model.split("/")[-1]
    except Exception:
        version = "latest"
    return ep.scoring_uri, key, version


def classify(image_bytes: bytes) -> dict:
    """Send one image to the endpoint and return its prediction dict."""
    scoring_uri, key, _ = get_endpoint()
    payload = {"image": base64.b64encode(image_bytes).decode("utf-8")}
    r = requests.post(scoring_uri, json=payload, timeout=60,
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"})
    r.raise_for_status()
    result = r.json()
    if isinstance(result, str):          # some servers return JSON as a string
        result = json.loads(result)
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])
    return result[0]


# ---------------------------------------------------------------- UI helpers

def wide(fn, *args, **kwargs):
    """Full-width element that works on both old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)

def result_card(res: dict):
    pred, conf = res["prediction"], res["confidence"]
    s = CLASS_STYLE.get(pred, {"color": "#333", "bg": "#eee", "icon": "", "note": ""})
    st.markdown(f"""
    <div class="result-card" style="background:{s['bg']}; border-left: 6px solid {s['color']};">
      <div class="result-label">Prediction</div>
      <div class="result-class" style="color:{s['color']}">{s['icon']} {pred}</div>
      <div class="result-conf">Confidence <b>{conf:.1%}</b> &nbsp;·&nbsp; {s['note']}</div>
    </div>""", unsafe_allow_html=True)
    if conf < LOW_CONFIDENCE:
        st.warning("Low confidence: the model is unsure about this image. "
                   "Check that it is a frontal chest X-ray, or ask a radiologist to review it.")


def probability_bars(probs: dict):
    st.markdown("**Probability for each class**")
    for cls in sorted(probs, key=probs.get, reverse=True):
        p = probs[cls]
        color = CLASS_STYLE.get(cls, {}).get("color", "#555")
        st.markdown(f"""
        <div class="bar-row">
          <div class="bar-head"><span>{cls}</span><span><b>{p:.1%}</b></span></div>
          <div class="bar-track"><div class="bar-fill" style="width:{p*100:.1f}%; background:{color};"></div></div>
        </div>""", unsafe_allow_html=True)


def sample_images():
    items = []
    if os.path.isdir(SAMPLE_DIR):
        for c in sorted(os.listdir(SAMPLE_DIR)):
            folder = os.path.join(SAMPLE_DIR, c)
            for f in sorted(os.listdir(folder))[:3]:
                items.append((f"{c} / {f}", os.path.join(folder, f), c))
    return items


if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🩻 X-Ray Insight")
    st.caption("Chest X-ray classifier on Azure Machine Learning")
    try:
        _, _, ver = get_endpoint()
        st.success(f"Endpoint online\n\n`{ENDPOINT_NAME}` · model v{ver}")
        online = True
    except Exception as e:
        st.error(f"Endpoint not reachable.\n\n{e}")
        online = False
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("ResNet-50 (transfer learning)  \nInput 224×224 RGB  \nClasses: Covid · Normal · Viral Pneumonia")
    st.markdown("**Test-set results**")
    c1, c2 = st.columns(2)
    c1.markdown('<div class="metric-box"><b>93.9%</b><span>accuracy</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="metric-box"><b>96.2%</b><span>Covid recall</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p class="disclaimer">Educational project. Not a medical device and not '
                'a substitute for a doctor\'s diagnosis.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- main
st.markdown('<div class="hero"><h1>COVID-19 Chest X-ray Classifier</h1>'
            '<p>Upload a chest X-ray and the model deployed on Azure ML will classify it as '
            '<b>Covid</b>, <b>Normal</b> or <b>Viral Pneumonia</b>.</p></div>', unsafe_allow_html=True)

tab_predict, tab_history, tab_model = st.tabs(["🔍 Analyse X-ray", "🕘 History", "📊 About the model"])

with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")
    image_bytes, image_name, true_label = None, None, None

    with left:
        source = st.radio("Image source", ["Upload my own X-ray", "Use a sample from the test set"],
                          horizontal=True, label_visibility="collapsed")
        if source.startswith("Upload"):
            up = st.file_uploader("Drop a chest X-ray here (JPG or PNG)", type=["jpg", "jpeg", "png"])
            if up is not None:
                image_bytes, image_name = up.getvalue(), up.name
        else:
            samples = sample_images()
            if samples:
                label = st.selectbox("Choose a test image", [s[0] for s in samples])
                _, path, true_label = next(s for s in samples if s[0] == label)
                with open(path, "rb") as f:
                    image_bytes = f.read()
                image_name = label
            else:
                st.info("Sample folder not found. Run the app from the covid-xray folder.")

        if image_bytes:
            wide(st.image, image_bytes, caption=image_name)

    with right:
        if not image_bytes:
            st.markdown("#### How it works")
            st.markdown(
                "1. **Upload** a frontal chest X-ray (JPG/PNG).\n"
                "2. The image is sent securely to the **Azure ML endpoint**.\n"
                "3. The ResNet-50 model returns the **class and confidence**.\n"
                "4. Results are saved in **History** for this session.")
            st.info("Tip: use *Use a sample from the test set* to try it instantly.")
        else:
            go = wide(st.button, "Analyse X-ray", type="primary", disabled=not online)
            if go:
                try:
                    t0 = time.time()
                    with st.spinner("Sending to Azure ML and running ResNet-50..."):
                        res = classify(image_bytes)
                    res["latency"] = time.time() - t0
                    st.session_state.last = {"name": image_name, "res": res, "true": true_label}
                    st.session_state.history.insert(0, {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Image": image_name,
                        "Prediction": res["prediction"],
                        "Confidence": f"{res['confidence']:.1%}",
                        "Actual (sample only)": true_label or "-",
                    })
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

            last = st.session_state.get("last")
            if last and last["name"] == image_name:
                result_card(last["res"])
                probability_bars(last["res"]["probabilities"])
                meta = f"Response time {last['res']['latency']:.1f}s from `{ENDPOINT_NAME}`"
                if last["true"]:
                    ok = last["true"] == last["res"]["prediction"]
                    meta += f" · Actual label: **{last['true']}** {'✅ correct' if ok else '❌ incorrect'}"
                st.caption(meta)

with tab_history:
    if st.session_state.history:
        wide(st.dataframe, st.session_state.history, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Analyse an X-ray and it will appear here.")

with tab_model:
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lab in [(m1, "93.94%", "Test accuracy (62/66)"), (m2, "251", "Training images"),
                          (m3, "66", "Test images"), (m4, "ResNet-50", "Architecture")]:
        col.markdown(f'<div class="metric-box"><b>{val}</b><span>{lab}</span></div>', unsafe_allow_html=True)
    st.markdown("#### Confusion matrix (test set)")
    st.table({"Actual \\ Predicted": CLASSES,
              "Covid": [25, 0, 0], "Normal": [1, 20, 3], "Viral Pneumonia": [0, 0, 17]})
    g1, g2 = st.columns(2)
    if os.path.exists("outputs/training_curves.png"):
        g1.image("outputs/training_curves.png", caption="Training vs validation curves")
    if os.path.exists("outputs/confusion_matrix.png"):
        g2.image("outputs/confusion_matrix.png", caption="Confusion matrix")
    st.markdown("#### Pipeline")
    st.markdown("Dataset → preprocessing & augmentation → ResNet-50 transfer learning (15 epochs) → "
                "evaluation → Azure ML model registry → managed online endpoint → this app")
