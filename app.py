"""
CRISIS-FUSION: Multimodal AI Disaster Intelligence & Management Dashboard
Final-Year Project: Disaster Management and Recommendation System
Dataset: CrisisMMD v2.0 | Framework: PyTorch, Transformers, Streamlit, Plotly
Author: SAMARTH SINHA (GitHub: @samarthsinha07)
"""

import os
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import model_engine as me

# ==============================================================================
# 1. PAGE CONFIG & CUSTOM RESEARCH AESTHETICS (CSS)
# ==============================================================================
st.set_page_config(
    page_title="Crisis-Fusion | Disaster Intelligence Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern Academic / Crisis Response Theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background & main containers */
.stApp {
    background: radial-gradient(circle at 15% 15%, #111827 0%, #0a0e17 100%);
    color: #f1f5f9;
}

/* Glassmorphic Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(10px);
    transition: all 0.25s ease-in-out;
}
.metric-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-2px);
}

.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}

.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

.metric-sub {
    font-size: 0.85rem;
    color: #38bdf8;
    margin-top: 0.25rem;
    font-weight: 500;
}

/* Badges */
.badge-severe {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid #ef4444;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-info {
    background: rgba(56, 189, 248, 0.2);
    color: #38bdf8;
    border: 1px solid #38bdf8;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-success {
    background: rgba(52, 211, 153, 0.2);
    color: #34d399;
    border: 1px solid #34d399;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-warning {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
    border: 1px solid #fbbf24;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

/* Monospace text */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(148, 163, 184, 0.1);
}

/* Header style */
.header-box {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    margin-bottom: 1.5rem;
}
.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.header-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.4rem;
}

/* Alert boxes */
.info-callout {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #38bdf8;
    padding: 1rem 1.25rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1.25rem;
}

.warning-callout {
    background: rgba(245, 158, 11, 0.1);
    border-left: 4px solid #f59e0b;
    padding: 1rem 1.25rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1.25rem;
}

.danger-callout {
    background: rgba(239, 68, 68, 0.1);
    border-left: 4px solid #ef4444;
    padding: 1rem 1.25rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1.25rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State for Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================================================================
# 2. SIDEBAR NAVIGATION & SYSTEM TELEMETRY
# ==============================================================================
with st.sidebar:
    st.markdown("### 🚨 CRISIS-FUSION AI")
    st.caption("Multimodal Disaster Intelligence System")
    st.markdown("---")

    menu_options = [
        "📊 Overview & Telemetry",
        "🔮 Live Multimodal Prediction",
        "🖼️ Vision & Image Modality Analysis",
        "📈 Model Evaluation & Benchmarks",
        "⚠️ Damage Severity Analysis (Task 3)",
        "📑 Academic Research & Paper Comparison",
    ]
    selected_page = st.radio("Navigation", menu_options, index=0)

    st.markdown("---")
    st.markdown("#### ⚙️ Runtime Diagnostics")

    # Real-time scan of checkpoint repository
    ckpts = me.scan_checkpoints()
    avail_ckpts = [v for v in ckpts.values() if v["exists"]]
    total_ckpts = len(ckpts)

    st.markdown(f"**Hardware Device:** `{me.DEVICE}`")
    st.markdown(f"**Dataset Status:** `CrisisMMD v2.0 Ready`")
    st.markdown(f"**Checkpoints Online:** `{len(avail_ckpts)} / {total_ckpts}`")

    with st.expander("🔍 Registered Checkpoint Files"):
        ckpt_df = pd.DataFrame([
            {
                "Task": v["task"],
                "Model": v["model"],
                "File": v["filename"],
                "Status": "🟢 Available" if v["exists"] else "⚪ Not on disk",
                "Size (MB)": v["size_mb"] if v["exists"] else "—",
            }
            for v in ckpts.values()
        ])
        st.dataframe(ckpt_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Final Year Capstone Project\n\nAuthor: **Samarth Sinha**\n\nGitHub: [@samarthsinha07](https://github.com/samarthsinha07)")


# ==============================================================================
# PAGE 1: OVERVIEW & SYSTEM TELEMETRY
# ==============================================================================
if selected_page == "📊 Overview & Telemetry":
    st.markdown(
        """
        <div class="header-box">
            <h1 class="header-title">🌍 Disaster Assessment Intelligence Overview</h1>
            <p class="header-subtitle">
                Multimodal Deep Learning for Real-Time Crisis Post Classification using the CrisisMMD v2.0 Benchmark
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Test Corpus Size</div>
                <div class="metric-value">4,552</div>
                <div class="metric-sub">Paired text-image tweets</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Task 1a Informativeness</div>
                <div class="metric-value">91.13%</div>
                <div class="metric-sub">+4.11 pp over best unimodal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Task 1b Disaster Type</div>
                <div class="metric-value">90.29%</div>
                <div class="metric-sub">+3.72 pp over best unimodal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Task 2 Humanitarian</div>
                <div class="metric-value">86.28%</div>
                <div class="metric-sub">+3.98 pp over best unimodal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Task 3 Damage Severity</div>
                <div class="metric-value">64.27%</div>
                <div class="metric-sub">ViT unimodal (class imbalanced)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Charts Section
    col_chart1, col_chart2 = st.columns([1.2, 1])

    with col_chart1:
        st.subheader("📊 Score Fusion vs. Best Unimodal Accuracy")
        chart_data = pd.DataFrame({
            "Task": [
                "Task 1a (Informativeness)",
                "Task 1b (Disaster Type)",
                "Task 2 (Humanitarian)",
                "Task 3 (Damage Severity)",
            ],
            "Best Unimodal": [87.03, 86.57, 82.30, 64.27],
            "Score Fusion (50/50)": [91.13, 90.29, 86.28, 63.14],
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=chart_data["Task"],
            y=chart_data["Best Unimodal"],
            name="Best Unimodal Classifier",
            marker_color="#64748b",
            text=[f"{v:.1f}%" for v in chart_data["Best Unimodal"]],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            x=chart_data["Task"],
            y=chart_data["Score Fusion (50/50)"],
            name="Multimodal Score Fusion",
            marker_color="#38bdf8",
            text=[f"{v:.1f}%" for v in chart_data["Score Fusion (50/50)"]],
            textposition="auto",
        ))
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            yaxis=dict(title="Accuracy (%)", range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.subheader("⚡ Fusion Accuracy Improvement (Delta)")
        delta_data = pd.DataFrame({
            "Task": ["Task 1a", "Task 1b", "Task 2", "Task 3"],
            "Delta (pp)": [4.11, 3.72, 3.98, -1.13],
            "Color": ["#34d399", "#34d399", "#34d399", "#f87171"],
        })
        fig_delta = px.bar(
            delta_data,
            x="Task",
            y="Delta (pp)",
            text="Delta (pp)",
            color="Delta (pp)",
            color_continuous_scale=["#f87171", "#fbbf24", "#34d399"],
            template="plotly_dark",
        )
        fig_delta.update_traces(texttemplate="%{text:+.2f} pp", textposition="outside")
        fig_delta.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            yaxis=dict(title="Accuracy Delta (Percentage Points)", range=[-3, 6]),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(fig_delta, use_container_width=True)

    # Dataset & Processed Stats
    st.markdown("### 🗄️ Dataset Partitioning & Class Coverage")
    if os.path.exists("data/processed/stats.csv"):
        stats_df = pd.read_csv("data/processed/stats.csv")
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python prepare_data.py` to inspect processed dataset splits.")

    # Recent Live Predictions Table
    st.markdown("### 🕒 Recent Live Inference Log")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No inference runs recorded yet in this session. Go to 'Live Multimodal Prediction' to test.")


# ==============================================================================
# PAGE 2: LIVE MULTIMODAL PREDICTION
# ==============================================================================
elif selected_page == "🔮 Live Multimodal Prediction":
    st.markdown(
        """
        <div class="header-box">
            <h1 class="header-title">🔮 Real-Time Multimodal Disaster Inference</h1>
            <p class="header-subtitle">
                Analyze paired social media reports (Tweet Text + Imagery) using fine-tuned transformer architectures
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset benchmark examples from CrisisMMD
    PRESETS = {
        "Preset 1: Hurricane Harvey Flooding (Informative / Infrastructure Damage)": {
            "task": "task1a",
            "text": "Hurricane Harvey batters residential streets; water levels rising rapidly in Houston neighborhood https://t.co/harveyfloods",
            "image_path": "data/CrisisMMD_v2.0/data_image/hurricane_maria/20_9_2017/910542719864397824_0.jpg",
        },
        "Preset 2: California Wildfires Smoke & Evacuation": {
            "task": "task1b",
            "text": "Wildfire warning issued for northern counties as CanyonFire spreads smoke and ash across interstate",
            "image_path": "data/CrisisMMD_v2.0/data_image/california_wildfires/10_10_2017/917804456422473734_0.jpg",
        },
        "Preset 3: Mexico Earthquake Rubble (Severe Damage)": {
            "task": "task3",
            "text": "Devastation from magnitude 7.1 earthquake in southern Mexico; buildings collapsed and rescue teams deployed",
            "image_path": "data/CrisisMMD_v2.0/data_image/hurricane_maria/27_9_2017/913116029156339713_0.jpg",
        },
        "Preset 4: Irrelevant Non-Disaster Social Post": {
            "task": "task1a",
            "text": "Cristofer Clemente Mora now in 2nd place at MontBlanc Marathon 80k race. Good luck everyone!",
            "image_path": "data/CrisisMMD_v2.0/data_image/srilanka_floods/23_6_2017/878185882431389696_0.jpg",
        },
    }

    selected_preset = st.selectbox(
        "⚡ Quick-Fill with CrisisMMD Benchmark Example:",
        ["None (Custom Input)"] + list(PRESETS.keys()),
    )

    default_text = ""
    default_img_path = None
    default_task = "task1a"

    if selected_preset != "None (Custom Input)":
        pdata = PRESETS[selected_preset]
        default_text = pdata["text"]
        default_img_path = pdata["image_path"]
        default_task = pdata["task"]

    col_config, col_inputs = st.columns([1, 2])

    with col_config:
        st.markdown("#### 🛠️ Task & Model Setup")
        task_choice = st.selectbox(
            "Classification Task",
            options=["task1a", "task1b", "task2", "task3"],
            format_func=lambda x: me.TASK_CONFIG[x]["title"],
            index=["task1a", "task1b", "task2", "task3"].index(default_task),
        )
        st.caption(me.TASK_CONFIG[task_choice]["description"])

        fusion_mode = st.radio(
            "Inference Pipeline",
            ["Multimodal Score Fusion (Recommended)", "Text-Only Classifier", "Image-Only Classifier", "Cross-Modal Attention Model"],
            index=0,
        )

        col_t, col_i = st.columns(2)
        with col_t:
            text_model_name = st.selectbox(
                "Text Backbone",
                options=["bert", "roberta", "bertweet"],
                index=0 if task_choice == "task1a" else 1,
            )
        with col_i:
            image_model_name = st.selectbox(
                "Vision Backbone",
                options=["resnet50", "vit", "densenet121"],
                index=0 if task_choice == "task1a" else 1,
            )

        if "Score Fusion" in fusion_mode:
            fusion_weight = st.slider(
                "Text Weight vs. Image Weight (α)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="P_fused = α * P_text + (1 - α) * P_image. Default is equal 0.5/0.5 weighting.",
            )
            st.caption(f"Score Formula: `{fusion_weight:.2f} × Text + {1.0 - fusion_weight:.2f} × Image`")
        else:
            fusion_weight = 0.5

    with col_inputs:
        st.markdown("#### 📝 Report Inputs")
        tweet_input = st.text_area(
            "Disaster Tweet / Social Post Content",
            value=default_text,
            height=110,
            placeholder="Type or paste the disaster-related tweet text here...",
        )

        uploaded_img = st.file_uploader(
            "Upload Disaster Image (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp"],
        )

        pil_img = None
        if uploaded_img is not None:
            pil_img = Image.open(uploaded_img)
            st.image(pil_img, caption="Uploaded Disaster Photo", width=300)
        elif default_img_path and os.path.exists(default_img_path):
            pil_img = Image.open(default_img_path)
            st.image(pil_img, caption=f"Loaded Preset: {os.path.basename(default_img_path)}", width=300)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚨 Analyze Disaster Report", type="primary", use_container_width=True)

    if analyze_btn:
        # Input validation
        if not tweet_input.strip() and "Text-Only" in fusion_mode:
            st.error("Please enter tweet text to perform text classification.")
            st.stop()
        if pil_img is None and "Image-Only" in fusion_mode:
            st.error("Please upload or select an image to perform image classification.")
            st.stop()
        if (not tweet_input.strip() or pil_img is None) and "Fusion" in fusion_mode:
            st.warning("⚠️ Multimodal fusion requires both text and image modalities. Please provide both inputs.")
            st.stop()

        # Checkpoint validation
        text_ckpt = f"best_{task_choice}_{text_model_name}.pt"
        image_ckpt = f"best_{task_choice}_{image_model_name}.pt"
        cross_ckpt = f"best_{task_choice}_fusion.pt"

        st.markdown("---")
        st.subheader("🔍 Prediction Results & Confidence Breakdown")

        try:
            start_time = time.time()

            if fusion_mode == "Text-Only Classifier":
                if not os.path.exists(text_ckpt):
                    st.error(
                        f"❌ **Missing Checkpoint:** `{text_ckpt}` was not found.\n\n"
                        f"To train this model on `{task_choice}`, run:\n"
                        f"```bash\npython baselines.py --task {task_choice} --model {text_model_name}\n```"
                    )
                    st.stop()
                probs = me.predict_text(task_choice, text_model_name, tweet_input, text_ckpt)
                text_probs = probs
                image_probs = None
                fused_probs = probs

            elif fusion_mode == "Image-Only Classifier":
                if not os.path.exists(image_ckpt):
                    st.error(
                        f"❌ **Missing Checkpoint:** `{image_ckpt}` was not found.\n\n"
                        f"To train this model on `{task_choice}`, run:\n"
                        f"```bash\npython baselines.py --task {task_choice} --model {image_model_name}\n```"
                    )
                    st.stop()
                probs = me.predict_image(task_choice, image_model_name, pil_img, image_ckpt)
                text_probs = None
                image_probs = probs
                fused_probs = probs

            elif fusion_mode == "Cross-Modal Attention Model":
                if not os.path.exists(cross_ckpt):
                    st.error(
                        f"❌ **Missing Checkpoint:** `{cross_ckpt}` was not found.\n\n"
                        f"To train the cross-modal attention fusion model on `{task_choice}`, run:\n"
                        f"```bash\npython fusion_model.py --task {task_choice}\n```"
                    )
                    st.stop()
                probs = me.predict_cross_modal(task_choice, tweet_input, pil_img, cross_ckpt)
                text_probs = None
                image_probs = None
                fused_probs = probs

            else:  # Multimodal Score Fusion
                # Verify checkpoints
                missing = []
                if not os.path.exists(text_ckpt):
                    missing.append(f"Text checkpoint: `{text_ckpt}`")
                if not os.path.exists(image_ckpt):
                    missing.append(f"Image checkpoint: `{image_ckpt}`")

                if missing:
                    st.error(
                        f"❌ **Required Checkpoint(s) Not Found:**\n\n"
                        + "\n".join([f"- {m}" for m in missing])
                        + f"\n\nTo train the missing models, run:\n"
                        f"```bash\npython baselines.py --task {task_choice} --model {text_model_name}\npython baselines.py --task {task_choice} --model {image_model_name}\n```"
                    )
                    st.stop()

                fused_probs, text_probs, image_probs = me.predict_score_fusion(
                    task=task_choice,
                    text_model_name=text_model_name,
                    image_model_name=image_model_name,
                    text=tweet_input,
                    image=pil_img,
                    weight=fusion_weight,
                    text_ckpt=text_ckpt,
                    image_ckpt=image_ckpt,
                )

            elapsed = round(time.time() - start_time, 3)

            # Determine winner class
            winner_class = max(fused_probs, key=fused_probs.get)
            winner_conf = fused_probs[winner_class] * 100

            # Display top banner
            badge_class = "badge-severe" if "severe" in winner_class or "earthquake" in winner_class else "badge-info"
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 6px solid #38bdf8; margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div class="metric-title">Predicted Classification</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #f8fafc; text-transform: uppercase;">
                                {winner_class.replace('_', ' ')}
                            </div>
                            <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.2rem;">
                                Confidence: <strong style="color: #38bdf8;">{winner_conf:.2f}%</strong> | Inference Latency: <code>{elapsed}s</code>
                            </div>
                        </div>
                        <div>
                            <span class="{badge_class}">Active Decision</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive Plotly probability distribution
            col_bar, col_mod = st.columns([1.3, 1])

            with col_bar:
                st.markdown("##### 📊 Fused Class Probability Distribution")
                df_probs = pd.DataFrame({
                    "Class": list(fused_probs.keys()),
                    "Probability": list(fused_probs.values()),
                }).sort_values(by="Probability", ascending=True)

                fig_p = px.bar(
                    df_probs,
                    x="Probability",
                    y="Class",
                    orientation="h",
                    text=[f"{p * 100:.1f}%" for p in df_probs["Probability"]],
                    template="plotly_dark",
                    color="Probability",
                    color_continuous_scale="Tealgrn",
                )
                fig_p.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    xaxis=dict(range=[0, 1.05], tickformat=".0%"),
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            with col_mod:
                st.markdown("##### 🤝 Modality Agreement Analysis")
                if text_probs and image_probs:
                    text_top = max(text_probs, key=text_probs.get)
                    image_top = max(image_probs, key=image_probs.get)
                    agreed = (text_top == image_top)

                    st.markdown(f"**Text Modality Top:** `{text_top}` ({text_probs[text_top]*100:.1f}%)")
                    st.markdown(f"**Image Modality Top:** `{image_top}` ({image_probs[image_top]*100:.1f}%)")
                    if agreed:
                        st.markdown("<span class='badge-success'>✓ Modalities in Full Agreement</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-warning'>⚠ Modalities Point to Different Classes</span>", unsafe_allow_html=True)
                        st.caption("Score fusion harmonized both signals to yield the consensus prediction above.")
                else:
                    st.info("Single modality or cross-modal end-to-end forward pass executed.")

            # Record in session state history
            st.session_state.history.append({
                "Timestamp": time.strftime("%H:%M:%S"),
                "Task": task_choice,
                "Model Pipeline": fusion_mode,
                "Top Prediction": winner_class,
                "Confidence (%)": round(winner_conf, 2),
                "Latency (s)": elapsed,
            })

        except Exception as e:
            st.error(f"Inference error: {str(e)}")


# ==============================================================================
# PAGE 3: VISION & IMAGE MODALITY ANALYSIS
# ==============================================================================
elif selected_page == "🖼️ Vision & Image Modality Analysis":
    st.markdown(
        """
        <div class="header-box">
            <h1 class="header-title">🖼️ Deep Vision & Image Modality Analysis</h1>
            <p class="header-subtitle">
                Inspect visual feature extraction, unimodal vision predictions, and comparative fusion behavior
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-callout">
            <strong>Honest Scientific Disclosure:</strong> The models in this repository perform <em>whole-image classification</em> 
            using Vision Transformers (ViT) and Convolutional Networks (ResNet-50 / DenseNet-121). 
            Fine-grained object bounding-box detection and Grad-CAM localization are <strong>not</strong> part of the trained classification heads 
            and are not fabricated.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_up, col_info = st.columns([1, 1.2])

    with col_up:
        img_file = st.file_uploader("Upload Image for Visual Inspection", type=["jpg", "png", "jpeg"])
        sample_choice = st.selectbox(
            "Or Pick a CrisisMMD Ground-Truth Sample:",
            [
                "None",
                "Hurricane Maria Flooding (Puerto Rico)",
                "California Wildfires (Smoke Column)",
                "Sri Lanka Floods (Submerged Road)",
            ],
        )

        img_to_show = None
        if img_file is not None:
            img_to_show = Image.open(img_file)
        elif sample_choice == "Hurricane Maria Flooding (Puerto Rico)":
            p = "data/CrisisMMD_v2.0/data_image/hurricane_maria/20_9_2017/910542719864397824_0.jpg"
            if os.path.exists(p):
                img_to_show = Image.open(p)
        elif sample_choice == "California Wildfires (Smoke Column)":
            p = "data/CrisisMMD_v2.0/data_image/california_wildfires/10_10_2017/917804456422473734_0.jpg"
            if os.path.exists(p):
                img_to_show = Image.open(p)
        elif sample_choice == "Sri Lanka Floods (Submerged Road)":
            p = "data/CrisisMMD_v2.0/data_image/srilanka_floods/23_6_2017/878185882431389696_0.jpg"
            if os.path.exists(p):
                img_to_show = Image.open(p)

        if img_to_show is not None:
            st.image(img_to_show, caption="Selected Image Modality", use_container_width=True)

    with col_info:
        if img_to_show is not None:
            st.markdown("#### 📐 Image Metadata & Preprocessing")
            w, h = img_to_show.size
            st.markdown(f"- **Raw Resolution:** `{w} × {h}` px")
            st.markdown(f"- **Aspect Ratio:** `{w/h:.2f} : 1`")
            st.markdown(f"- **Color Channels:** `{img_to_show.mode}` (Converted to 3-channel RGB)")
            st.markdown("- **Model Input Transform:** `Resize(256) ➔ CenterCrop(224) ➔ ImageNet Normalization`")

            st.markdown("#### 🤖 Image-Only Model Prediction")
            v_task = st.selectbox("Task to Evaluate on Image", ["task1a", "task1b", "task2", "task3"], format_func=lambda x: me.TASK_CONFIG[x]["title"])
            v_model = st.selectbox("Vision Backbone", ["resnet50", "vit", "densenet121"])

            v_ckpt = f"best_{v_task}_{v_model}.pt"
            if os.path.exists(v_ckpt):
                if st.button("Run Image-Only Inference", type="primary"):
                    with st.spinner("Processing image tensor..."):
                        i_probs = me.predict_image(v_task, v_model, img_to_show, v_ckpt)
                        top_c = max(i_probs, key=i_probs.get)
                        st.success(f"**Top Image Prediction:** `{top_c}` ({i_probs[top_c]*100:.1f}%)")

                        df_p = pd.DataFrame({"Class": list(i_probs.keys()), "Probability": list(i_probs.values())})
                        fig = px.bar(df_p, x="Probability", y="Class", orientation="h", template="plotly_dark")
                        fig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Checkpoint `{v_ckpt}` is not present on disk. Place it in the root folder or train via `python baselines.py --task {v_task} --model {v_model}`.")
        else:
            st.info("Upload an image or pick a sample to inspect its visual representations.")


# ==============================================================================
# PAGE 4: MODEL EVALUATION & BENCHMARKS
# ==============================================================================
elif selected_page == "📈 Model Evaluation & Benchmarks":
    st.markdown(
        """
        <div class="header-box">
            <h1 class="header-title">📈 Model Evaluation, Benchmarks & Metrics</h1>
            <p class="header-subtitle">
                Rigorous empirical evaluation comparing Unimodal Baselines, Multimodal Score Fusion, and Cross-Modal Attention
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_eval1, tab_eval2, tab_eval3, tab_eval4 = st.tabs([
        "📊 Comprehensive Results Table",
        "🔲 Confusion Matrices",
        "🔬 Task 1a Ablation Study",
        "💾 Export Evaluation CSVs",
    ])

    with tab_eval1:
        st.subheader("Comprehensive Experimental Evaluation Summary")
        if os.path.exists("figures/phase5_final_results.csv"):
            res_df = pd.read_csv("figures/phase5_final_results.csv")
            st.dataframe(res_df.style.highlight_max(subset=["Accuracy_%", "Weighted_F1_%"], color="#1e3a5f"), use_container_width=True)
        elif os.path.exists("results.csv"):
            res_df = pd.read_csv("results.csv")
            st.dataframe(res_df, use_container_width=True)

        st.markdown("#### 📊 Task-Wise Unimodal vs. Multimodal Comparison")
        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("figures/report/accuracy_comparison.png"):
                st.image("figures/report/accuracy_comparison.png", caption="Accuracy Comparison across Tasks", use_container_width=True)
        with c2:
            if os.path.exists("figures/report/f1_comparison.png"):
                st.image("figures/report/f1_comparison.png", caption="Weighted F1 Score Comparison", use_container_width=True)

    with tab_eval2:
        st.subheader("Official Evaluation Confusion Matrices")
        st.caption("Empirical confusion matrices computed on test splits for multimodal score fusion.")

        cm_task = st.selectbox("Select Task Confusion Matrix", ["Task 1b (Disaster Type)", "Task 2 (Humanitarian)", "Task 3 (Damage Severity)"])

        if "Task 1b" in cm_task:
            cm_path = "figures/task1b_score_fusion_confusion_matrix.csv"
            meta_path = "figures/task1b_score_fusion_metrics.json"
        elif "Task 2" in cm_task:
            cm_path = "figures/task2_score_fusion_confusion_matrix.csv"
            meta_path = "figures/task2_score_fusion_metrics.json"
        else:
            cm_path = "figures/task3_score_fusion_confusion_matrix.csv"
            meta_path = "figures/task3_score_fusion_metrics.json"

        if os.path.exists(cm_path):
            cm_df = pd.read_csv(cm_path, index_col=0)
            fig_cm = px.imshow(
                cm_df.values,
                labels=dict(x="Predicted Class", y="True Ground-Truth Class", color="Count"),
                x=list(cm_df.columns),
                y=list(cm_df.index),
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig_cm.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metrics_data = json.load(f)
            st.markdown("##### Task Evaluation Metrics")
            m_cols = st.columns(4)
            m_cols[0].metric("Accuracy", f"{metrics_data.get('acc', 0)*100:.2f}%")
            m_cols[1].metric("Weighted F1", f"{metrics_data.get('weighted_f1', 0)*100:.2f}%")
            m_cols[2].metric("Macro F1", f"{metrics_data.get('macro_f1', 0)*100:.2f}%")
            m_cols[3].metric("MCC", f"{metrics_data.get('mcc', 0):.4f}")

    with tab_eval3:
        st.subheader("🔬 Task 1a Ablation Study: Dissecting Multimodal Fusion")
        st.markdown(
            """
            To measure the exact contribution of each architectural component, an ablation study was conducted on Task 1a:
            - **Full Architecture:** RoBERTa + ViT + Cross-Modal Attention + Self-Attention
            - **No Cross-Modal:** Omits cross-attention, concatenating unimodal embeddings directly
            - **No Self-Attention:** Retains cross-attention but removes the self-attention refinement layer
            - **Unimodal Baselines:** Text-only (RoBERTa) vs. Image-only (ViT)
            """
        )
        if os.path.exists("ablation_results.csv"):
            ab_df = pd.read_csv("ablation_results.csv")
            st.dataframe(ab_df.style.highlight_max(subset=["accuracy", "f1", "auc_roc"], color="#1e3a5f"), use_container_width=True)

            fig_ab = px.bar(
                ab_df,
                x="ablation",
                y="accuracy",
                color="f1",
                text=[f"{v*100:.2f}%" for v in ab_df["accuracy"]],
                template="plotly_dark",
                color_continuous_scale="Viridis",
                title="Task 1a Ablation: Classification Accuracy by Variant",
            )
            fig_ab.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350)
            st.plotly_chart(fig_ab, use_container_width=True)

    with tab_eval4:
        st.subheader("💾 Download Benchmark Evaluation Data (CSV)")
        st.caption("Directly export the experiment logs and evaluation matrices for your research papers or reports.")

        files_to_download = [
            ("Phase 5 Final Clean Summary", "figures/phase5_final_results.csv"),
            ("Comprehensive Results Matrix", "results.csv"),
            ("Ablation Study Results", "ablation_results.csv"),
            ("Task 3 Imbalance Improvement Experiments", "task3_improvement_results.csv"),
        ]

        for label, fpath in files_to_download:
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {label} ({os.path.basename(fpath)})",
                        data=f,
                        file_name=os.path.basename(fpath),
                        mime="text/csv",
                    )


# ==============================================================================
# PAGE 5: DAMAGE SEVERITY ANALYSIS (TASK 3 DEEP DIVE)
# ==============================================================================
elif selected_page == "⚠️ Damage Severity Analysis (Task 3)":
    st.markdown(
        """
        <div class="header-box">
            <h1 class="header-title">⚠️ Damage Severity Assessment (Task 3 Deep Dive)</h1>
            <p class="header-subtitle">
                In-depth analysis of physical infrastructure damage classification, extreme class imbalance, and mitigation strategies
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="warning-callout">
            <strong>Key Finding on Task 3:</strong> While multimodal fusion dramatically boosted Tasks 1a, 1b, and 2, 
            Task 3 remains the most challenging task in CrisisMMD due to severe class imbalance 
            (over 62% of samples belong to the 'Severe Damage' class). Unimodal ViT achieved 64.27% accuracy, 
            while standard score fusion scored 63.14%.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_t3_1, col_t3_2 = st.columns([1, 1.2])

    with col_t3_1:
        st.markdown("#### 📊 Task 3 Class Imbalance in Test Split")
        t3_dist = pd.DataFrame({
            "Severity Class": ["Severe damage", "Mild damage", "Little or no damage"],
            "Sample Count": [332, 126, 71],
            "Percentage": ["62.8%", "23.8%", "13.4%"],
        })
        st.dataframe(t3_dist, use_container_width=True, hide_index=True)

        fig_pie = px.pie(
            t3_dist,
            names="Severity Class",
            values="Sample Count",
            color="Severity Class",
            color_discrete_map={
                "Severe damage": "#ef4444",
                "Mild damage": "#f59e0b",
                "Little or no damage": "#10b981",
            },
            template="plotly_dark",
            hole=0.45,
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_t3_2:
        st.markdown("#### 🔬 Improvement Experiments (Handling Imbalance)")
        st.caption("Comparison of techniques implemented in `improve_task3.py` to boost minority class recall.")

        if os.path.exists("task3_improvement_results.csv"):
            imp_df = pd.read_csv("task3_improvement_results.csv")
            display_cols = ["class_weights", "augment", "oversample", "acc", "macro_f1", "weighted_f1", "balanced_accuracy"]
            clean_imp = imp_df[display_cols].copy()
            clean_imp["acc"] = (clean_imp["acc"] * 100).round(2).astype(str) + "%"
            clean_imp["macro_f1"] = (clean_imp["macro_f1"] * 100).round(2).astype(str) + "%"
            clean_imp["weighted_f1"] = (clean_imp["weighted_f1"] * 100).round(2).astype(str) + "%"
            st.dataframe(clean_imp, use_container_width=True, hide_index=True)

        if os.path.exists("figures/report/task3_improvements.png"):
            st.image("figures/report/task3_improvements.png", caption="Task 3 Imbalance Mitigation Tradeoffs", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 💡 Technical Insights on Disaster Damage Estimation")
    st.markdown(
        """
        1. **Dominance of Visual Information:** Unlike informativeness or humanitarian categories which rely heavily on text linguistics, 
           damage severity is inherently visual. Text like *"Devastation in Puerto Rico"* conveys emotional tone but lacks the granularity 
           to distinguish whether a wall has minor cracks (*mild damage*) or has collapsed entirely (*severe damage*).
        2. **Class-Weighted Loss vs. Oversampling:** Adding inverse-frequency class weights with image data augmentation increased **Macro F1 from 45.64% to 56.66%**, 
           meaning the model successfully learned to detect mild and minor damage rather than defaulting to the majority severe class.
        """
    )


# ==============================================================================
# PAGE 6: ACADEMIC RESEARCH & REFERENCE PAPER COMPARISON
# ==============================================================================
elif selected_page == "📑 Academic Research & Paper Comparison":
    st.markdown(
        """
        <div class="header-box">
            <h1 class="header-title">📑 Academic Research & Benchmark Comparison</h1>
            <p class="header-subtitle">
                Systematic alignment with published research: Shetty et al. (Springer, 2024/2025) vs. Experimental Project Results
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-callout">
            <strong>Research Context & Attribution:</strong><br>
            Reference Research Paper: <em>Disaster assessment from social media using multimodal deep learning</em><br>
            <strong>Authors:</strong> Nisha P. Shetty, Yash Bijalwan, Pranav Chaudhari, Jayashree Shetty, Balachandra Muniyal<br>
            <strong>Journal:</strong> <em>Multimedia Tools and Applications</em> (Springer, 2024/2025)<br>
            <strong>Implementation Repository:</strong> <code>Crisis-Fusion</code> by Samarth Sinha (@samarthsinha07)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚖️ Side-by-Side Benchmark Comparison")
    st.caption("Notice: Published literature benchmarks are transparently distinguished from the experimental figures achieved in this implementation.")

    comp_table = pd.DataFrame({
        "Disaster Assessment Task": [
            "Task 1a — Informativeness",
            "Task 1b — Disaster Type",
            "Task 2 — Humanitarian Info",
            "Task 3 — Damage Severity",
        ],
        "Reference Paper (Shetty et al. 2025)": [
            "91.53% Acc | 0.915 F1 (Middle Fusion)",
            "91.07% Acc | 0.910 F1 (Middle Fusion)",
            "89.63% Acc | 0.896 F1 (Middle Fusion)",
            "70.32% Acc | 0.687 F1 (Middle Fusion)",
        ],
        "Our Project — Score Fusion": [
            "91.13% Acc | 0.910 F1 (RoBERTa + ViT)",
            "90.29% Acc | 0.902 F1 (RoBERTa + ViT)",
            "86.28% Acc | 0.858 F1 (BERTweet + ViT)",
            "63.14% Acc | 0.494 F1 (RoBERTa + ViT)",
        ],
        "Our Project — Unimodal Best": [
            "87.03% (RoBERTa)",
            "86.57% (RoBERTa)",
            "82.30% (BERTweet)",
            "64.27% (ViT)",
        ],
        "Fusion Delta": [
            "+4.11 pp",
            "+3.72 pp",
            "+3.98 pp",
            "-1.13 pp",
        ],
    })
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏗️ Architectural Topology Comparison")

    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("#### 1. Reference Paper Proposed Middle Fusion")
        st.markdown(
            """
            - **Text Stream:** RoBERTa (`ht` in R^(128x1024))
            - **Image Stream:** ViT (`hi` in R^(197x768))
            - **Dimensional Projection:**
              - Text Encoder `Et`: Conv1d + Linear ➔ `ht_bar` in R^128
              - Image Encoder `Ei`: Conv1d + Linear ➔ `hi_bar` in R^128
              - Shared Encoder `Es`: Linear ➔ `h_shared` in R^128
            - **Cross-Modal Attention:** Generates cross-attended matrix `Xm` in R^(4x128)
            - **Multimodal Association:** 8-head Transformer self-attention layer
            - **Decoder:** 2 Linear layers with Tanh ➔ Output classes
            """
        )

    with col_arch2:
        st.markdown("#### 2. Implemented Score-Fusion Architecture")
        st.markdown(
            """
            - **Text Stream:** Fine-tuned Transformer (`BERT`, `RoBERTa-large`, or `BERTweet`)
            - **Image Stream:** Fine-tuned Vision Transformer (`google/vit-base-patch16-224-in21k`)
            - **Fusion Mechanism:**
              - Softmax logits extraction from both models
              - Equal probability weighted averaging: `P_fused = 0.5 * P_text + 0.5 * P_image`
            - **Cross-Modal Variant (`fusion_model.py`):**
              - RoBERTa-large + ViT-base
              - Linear projection to 512 dimensions
              - Bidirectional cross-attention + Transformer self-attention
              - Evaluated on Task 1a (89.50% Acc, 89.43% F1)
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚠️ Key Limitations & Directions for Future Work")
    st.markdown(
        """
        1. **Fixed vs. Learned Fusion Weights:** Current score fusion uses static 50/50 weights. Learning an adaptive gating network conditioned on post length and image quality would dynamically favor the more reliable modality.
        2. **Conflict Resolution:** When a disaster photo accompanies an unrelated tweet (or sarcastic caption), unimodal predictions conflict. A cross-modal relevance gate would reject mismatched pairs.
        3. **Real-Time Deployment:** Deploying full RoBERTa-large and ViT requires substantial memory. Exploring model quantization (INT8) or distilled student models (DistilBERT + MobileNetV3) would enable edge deployment for first responders.
        """
    )
