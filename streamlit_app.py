"""
SentNL — Streamlit UI

Interactive web interface for:
  • Task selection & text classification
  • Probability visualisation
  • Model comparison dashboard
  • Confusion matrix & feature importance viewers
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import ConfigManager
from src.tasks.spam.predictor import SpamPredictor
from src.tasks.sentiment.predictor import SentimentPredictor
from src.tasks.topics.predictor import TopicsPredictor
from src.registry import register_task, get_predictor_class, list_tasks

# ── Page Config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="SentNL — NLP Classification Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Register Tasks ────────────────────────────────────────────────────

register_task("spam", SpamPredictor)
register_task("sentiment", SentimentPredictor)
register_task("topics", TopicsPredictor)

config_manager = ConfigManager()

# ── Custom CSS ────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --accent: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text: #e2e8f0;
        --text-muted: #94a3b8;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    /* Hero section */
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .prediction-label {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .confidence-value {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        flex: 1;
        text-align: center;
    }

    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366f1;
    }

    .metric-card .label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">🧠 SentNL</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    "Unified NLP Classification Platform — Spam · Sentiment · Topics"
    "</div>",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    available_tasks = list_tasks()
    task_labels = {
        "spam": "🛡️ Spam Detection",
        "sentiment": "💬 Sentiment Analysis",
        "topics": "📂 Topic Classification",
    }

    selected_task = st.selectbox(
        "Select Task",
        available_tasks,
        format_func=lambda t: task_labels.get(t, t),
    )

    st.markdown("---")
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "View",
        ["🔍 Predict", "📈 Model Comparison", "📋 Experiment Log"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#64748b; font-size:0.8rem;'>"
        "Built with engineering maturity.<br>"
        "SentNL v1.0"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Prediction Page ──────────────────────────────────────────────────

if "🔍 Predict" in page:

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📝 Input Text")
        user_text = st.text_area(
            "Enter text to classify",
            height=150,
            placeholder={
                "spam": "e.g., Congratulations! You've won a free iPhone! Click here now!",
                "sentiment": "e.g., This movie was absolutely brilliant, a true masterpiece.",
                "topics": "e.g., The stock market saw significant gains today as tech shares rallied.",
            }.get(selected_task, "Enter your text here…"),
            label_visibility="collapsed",
        )

        predict_btn = st.button(
            "🚀 Classify",
            use_container_width=True,
            type="primary",
        )

    with col2:
        st.markdown("### 📋 Task Info")
        task_config = config_manager.get_task_config(selected_task)
        task_info = task_config.get("task", {})
        st.info(
            f"**{task_info.get('display_name', selected_task)}**\n\n"
            f"{task_info.get('description', '')}"
        )

        expected = task_config.get("dataset", {}).get("expected_labels", [])
        if expected:
            st.markdown("**Expected labels:**")
            for label in expected:
                st.markdown(f"  `{label}`")

    if predict_btn and user_text.strip():
        with st.spinner("Classifying…"):
            try:
                config = config_manager.get_task_config(selected_task)
                predictor_cls = get_predictor_class(selected_task)
                predictor = predictor_cls(PROJECT_ROOT, config)
                result = predictor.predict(user_text)

                pred = result["prediction"]
                label = pred["label"]
                confidence = pred["confidence"]
                probs = pred.get("probabilities", {})

                # ── Result Display ──
                st.markdown("---")

                # Colour by task
                label_colours = {
                    "spam": {"spam": "#ef4444", "ham": "#10b981"},
                    "sentiment": {"positive": "#10b981", "negative": "#ef4444"},
                }
                colour = label_colours.get(selected_task, {}).get(
                    label, "#6366f1"
                )

                r1, r2, r3 = st.columns([2, 2, 1])
                with r1:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div style="color:#94a3b8;font-size:0.85rem;">PREDICTION</div>'
                        f'<div class="prediction-label" style="color:{colour};">'
                        f"{label}</div></div>",
                        unsafe_allow_html=True,
                    )

                with r2:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div style="color:#94a3b8;font-size:0.85rem;">CONFIDENCE</div>'
                        f'<div class="confidence-value" style="color:{colour};">'
                        f"{confidence:.1%}</div></div>",
                        unsafe_allow_html=True,
                    )

                with r3:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div style="color:#94a3b8;font-size:0.85rem;">LATENCY</div>'
                        f'<div style="font-size:1.5rem;font-weight:600;color:#06b6d4;">'
                        f'{result["processing_time_ms"]:.1f}ms</div></div>',
                        unsafe_allow_html=True,
                    )

                # ── Probability Bar Chart ──
                if probs:
                    st.markdown("### 📊 Probability Distribution")
                    prob_df = pd.DataFrame(
                        list(probs.items()), columns=["Label", "Probability"]
                    ).sort_values("Probability", ascending=True)

                    colours = [
                        colour if l == label else "#475569"
                        for l in prob_df["Label"]
                    ]

                    fig = go.Figure(
                        go.Bar(
                            x=prob_df["Probability"],
                            y=prob_df["Label"],
                            orientation="h",
                            marker_color=colours,
                            text=[f"{p:.1%}" for p in prob_df["Probability"]],
                            textposition="auto",
                            textfont=dict(color="white", size=14),
                        )
                    )
                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e2e8f0"),
                        xaxis=dict(
                            range=[0, 1],
                            showgrid=True,
                            gridcolor="#334155",
                            tickformat=".0%",
                        ),
                        yaxis=dict(showgrid=False),
                        height=max(200, len(probs) * 60),
                        margin=dict(l=0, r=20, t=10, b=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Raw JSON
                with st.expander("🔧 Raw Response JSON"):
                    st.json(result)

            except FileNotFoundError:
                st.error(
                    f"⚠️ No trained model found for **{selected_task}**. "
                    f"Run `python main.py train {selected_task}` first."
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")

    elif predict_btn:
        st.warning("Please enter some text to classify.")


# ── Model Comparison Page ─────────────────────────────────────────────

elif "📈 Model Comparison" in page:
    st.markdown("### 📈 Model Comparison Dashboard")

    experiments_path = PROJECT_ROOT / "reports" / "experiments.csv"
    if not experiments_path.exists():
        st.info(
            "No experiments found yet. "
            "Run `python main.py train <task>` to generate data."
        )
    else:
        df = pd.read_csv(experiments_path)

        if selected_task:
            task_df = df[df["task"] == selected_task]
        else:
            task_df = df

        if task_df.empty:
            st.info(f"No experiments found for task: {selected_task}")
        else:
            # Metrics summary
            st.markdown("#### 🏆 Leaderboard")
            display_cols = [
                "model_name",
                "version",
                "accuracy",
                "macro_f1",
                "f1_weighted",
                "training_time_seconds",
                "inference_time_ms",
            ]
            available_cols = [c for c in display_cols if c in task_df.columns]
            leaderboard = (
                task_df[available_cols]
                .sort_values("macro_f1", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(leaderboard, use_container_width=True)

            # Accuracy comparison chart
            if len(task_df) > 1:
                st.markdown("#### 📊 Metric Comparison")

                metric_choice = st.selectbox(
                    "Select Metric",
                    ["accuracy", "macro_f1", "f1_weighted", "precision_weighted", "recall_weighted"],
                )

                fig = px.bar(
                    task_df.sort_values(metric_choice, ascending=False),
                    x="model_name",
                    y=metric_choice,
                    color="model_name",
                    text=metric_choice,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    showlegend=False,
                    yaxis=dict(range=[0, 1], gridcolor="#334155"),
                    xaxis=dict(showgrid=False),
                )
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            # Training time comparison
            st.markdown("#### ⏱️ Training & Inference Speed")
            speed_cols = st.columns(2)
            with speed_cols[0]:
                fig_train = px.bar(
                    task_df,
                    x="model_name",
                    y="training_time_seconds",
                    title="Training Time (seconds)",
                    color_discrete_sequence=["#6366f1"],
                )
                fig_train.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(fig_train, use_container_width=True)

            with speed_cols[1]:
                fig_inf = px.bar(
                    task_df,
                    x="model_name",
                    y="inference_time_ms",
                    title="Inference Time (ms)",
                    color_discrete_sequence=["#06b6d4"],
                )
                fig_inf.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                )
                st.plotly_chart(fig_inf, use_container_width=True)

            # Confusion matrix viewer
            st.markdown("#### 🔢 Confusion Matrix Viewer")
            versions = task_df["version"].tolist()
            selected_version = st.selectbox("Select run version", versions)
            cm_path = (
                PROJECT_ROOT
                / "reports"
                / selected_task
                / selected_version
                / "confusion_matrix.png"
            )
            if cm_path.exists():
                st.image(str(cm_path), caption=f"Confusion Matrix — {selected_version}")
            else:
                st.info("Confusion matrix not found for this run.")

            # Feature importance viewer
            st.markdown("#### 🔑 Feature Importance")
            fi_path = (
                PROJECT_ROOT
                / "reports"
                / selected_task
                / selected_version
                / "feature_importance.csv"
            )
            if fi_path.exists():
                fi_df = pd.read_csv(fi_path)
                for cls in fi_df["class"].unique():
                    class_df = fi_df[fi_df["class"] == cls].head(10)
                    fig_fi = px.bar(
                        class_df.sort_values("weight"),
                        x="weight",
                        y="feature",
                        orientation="h",
                        title=f"Top Features — {cls}",
                        color_discrete_sequence=["#10b981"],
                    )
                    fig_fi.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e2e8f0"),
                        height=350,
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance data not found for this run.")


# ── Experiment Log Page ───────────────────────────────────────────────

elif "📋 Experiment Log" in page:
    st.markdown("### 📋 Full Experiment History")

    experiments_path = PROJECT_ROOT / "reports" / "experiments.csv"
    if not experiments_path.exists():
        st.info("No experiments recorded yet.")
    else:
        df = pd.read_csv(experiments_path)
        st.dataframe(df, use_container_width=True)

        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download experiments.csv",
            data=csv_data,
            file_name="experiments.csv",
            mime="text/csv",
        )

        # Summary stats
        st.markdown("#### 📊 Summary Statistics")
        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.metric("Total Runs", len(df))
        with summary_cols[1]:
            st.metric("Tasks Covered", df["task"].nunique())
        with summary_cols[2]:
            st.metric(
                "Best Accuracy",
                f"{df['accuracy'].max():.4f}" if "accuracy" in df.columns else "N/A",
            )
