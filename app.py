import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SemantiQ – Text Similarity Analyzer", layout="wide", page_icon="🔥")

# ── Color Palette (Sunset Orange) ─────────────────────────────────────────────
# BG:       #0d0d0d  (near-black)
# Card:     #1a1108  (warm dark brown)
# Border:   #c2410c  (burnt orange)
# Accent1:  #f97316  (orange)
# Accent2:  #fbbf24  (amber/gold)
# Text:     #fef3c7  (warm cream)
# Muted:    #92400e  (dark amber)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0d;
}

/* Main background */
.stApp { background-color: #0d0d0d; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

/* Hero header banner */
.hero-banner {
    background: linear-gradient(135deg, #1a0f00 0%, #2d1600 50%, #1a0800 100%);
    border: 1px solid #c2410c;
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(249,115,22,0.15);
}
.hero-title {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #f97316, #fbbf24);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-sub { color: #d97706; font-size: 0.95rem; margin: 0; }

/* Section headings */
h2, h3, h4 { color: #fbbf24 !important; font-weight: 600 !important; }

/* Divider */
hr { border-color: #c2410c44; margin: 1.5rem 0; }

/* Text areas */
.stTextArea textarea {
    background: #1a1108 !important;
    color: #fef3c7 !important;
    border: 1.5px solid #92400e !important;
    border-radius: 12px !important;
    font-size: 0.9rem;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 2px rgba(249,115,22,0.2) !important;
}
.stTextArea label { color: #fbbf24 !important; font-weight: 600 !important; font-size: 0.85rem !important; }

/* Sliders */
.stSlider label { color: #fbbf24 !important; font-weight: 600 !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #f97316 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ea580c, #f97316, #fbbf24) !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.6rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em;
    transition: all 0.25s !important;
    box-shadow: 0 4px 15px rgba(249,115,22,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px rgba(249,115,22,0.5) !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1c1005, #251608);
    border: 1.5px solid #92400e;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.35rem 0;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover { border-color: #f97316; transform: translateY(-2px); }
.score-label { color: #f97316; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; }
.score-value { color: #fef3c7; font-size: 1.65rem; font-weight: 700; line-height: 1.2; margin: 0.2rem 0 0.1rem 0; }
.score-desc { color: #78350f; font-size: 0.73rem; line-height: 1.4; }

/* Paul card (standards) */
.paul-card {
    background: linear-gradient(145deg, #1c1005, #251608);
    border: 1.5px solid #92400e;
    border-left: 4px solid #f97316;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.35rem 0;
    transition: all 0.2s;
}
.paul-card:hover { border-left-color: #fbbf24; transform: translateX(3px); }

/* Section card wrapper */
.section-card {
    background: #130c02;
    border: 1px solid #92400e33;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #100a02 !important;
    border-right: 1px solid #92400e !important;
}
[data-testid="stSidebar"] h3 { color: #fbbf24 !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { color: #d97706 !important; }

/* Success / info / warning */
.stSuccess { background: #1a2e0a !important; border-color: #16a34a !important; border-radius: 10px !important; }
.stInfo { background: #1c1005 !important; border-color: #f97316 !important; border-radius: 10px !important; }
.stWarning { border-radius: 10px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Spinner */
.stSpinner > div { border-top-color: #f97316 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🔥 SemantiQ — Text Similarity Analyzer</div>
  <p class="hero-sub">Enter 2–10 texts · Cosine similarity via <code>all-MiniLM-L6-v2</code> · Scored against Paul's 7 Intellectual Standards</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔥 Paul's Standards")
    st.markdown("""
| | Standard | Measures |
|---|---|---|
| 🎯 | Clarity | Sentence structure & punctuation |
| ✅ | Accuracy | Semantic topic consistency |
| 📐 | Precision | Vocabulary richness |
| 🔗 | Relevance | Topical relatedness |
| 🧠 | Logic | Similarity consistency |
| ⭐ | Significance | Meaningful distinctions |
| ⚖️ | Fairness | Balanced comparisons |
""")
    st.markdown("---")
    st.markdown("**Model:** `all-MiniLM-L6-v2`")
    st.markdown("**Metric:** Cosine Similarity (0 → 1)")
    st.markdown("---")
    st.markdown("<span style='color:#78350f;font-size:0.8rem;'>🟠 High ≥ 70% &nbsp; 🟡 Mid ≥ 45% &nbsp; 🔴 Low < 45%</span>", unsafe_allow_html=True)

# ── Input Section ──────────────────────────────────────────────────────────────
st.markdown("### ✏️ Enter Your Texts")
n_texts = st.slider("Number of texts to compare", min_value=2, max_value=10, value=4)

texts = []
cols_per_row = 2
rows = (n_texts + 1) // cols_per_row

for row in range(rows):
    cols = st.columns(cols_per_row)
    for col_idx in range(cols_per_row):
        text_idx = row * cols_per_row + col_idx
        if text_idx < n_texts:
            with cols[col_idx]:
                t = st.text_area(
                    f"Text {text_idx + 1}",
                    placeholder=f"Enter sentence or paragraph #{text_idx + 1}...",
                    height=95,
                    key=f"text_{text_idx}"
                )
                texts.append(t)

# ── Control Row ────────────────────────────────────────────────────────────────
st.markdown("---")
col_k, col_btn, col_reset = st.columns([3, 1, 1])
with col_k:
    top_k = st.slider("Top-K similar pairs to show", min_value=1,
                      max_value=max(1, (n_texts * (n_texts - 1)) // 2), value=3)
with col_btn:
    analyse = st.button("🚀 Analyse")
with col_reset:
    if st.button("🔄 Reset"):
        st.rerun()

# ── Helper: Paul's Standards Computation ──────────────────────────────────────
def compute_paul_standards(texts, sim_matrix):
    scores = {}
    n = len(texts)
    off_diag = [sim_matrix[i][j] for i in range(n) for j in range(i+1, n)]
    avg_sim = np.mean(off_diag)

    # Clarity — sentence length + punctuation
    clarity_scores = []
    for t in texts:
        wc = len(t.split())
        if wc == 0:
            clarity_scores.append(0)
        else:
            opt = 1 - abs(wc - 15) / 50
            pb = 0.1 if any(c in t for c in ".!?,") else 0
            clarity_scores.append(max(0, min(1, opt + pb)))
    scores["Clarity"] = round(np.mean(clarity_scores) * 100, 1)

    # Accuracy — avg pairwise similarity (0.6 ideal)
    accuracy = 1 - abs(avg_sim - 0.6) / 0.6
    scores["Accuracy"] = round(max(0, min(1, accuracy)) * 100, 1)

    # Precision — vocabulary richness
    prec = []
    for t in texts:
        words = re.findall(r'\w+', t.lower())
        if not words:
            prec.append(0)
        else:
            ur = len(set(words)) / len(words)
            lb = min(len(words) / 30, 0.3)
            prec.append(min(1, ur * 0.8 + lb))
    scores["Precision"] = round(np.mean(prec) * 100, 1)

    # Relevance — max pairwise similarity
    scores["Relevance"] = round(max(off_diag) * 100, 1) if off_diag else 0

    # Logic — low variance = logical
    if len(off_diag) > 1:
        scores["Logic"] = round(max(0, 1 - np.var(off_diag) * 10) * 100, 1)
    else:
        scores["Logic"] = round(avg_sim * 100, 1)

    # Significance — spread between max and min
    if len(off_diag) > 1:
        scores["Significance"] = round(min(1, (max(off_diag) - min(off_diag)) * 2) * 100, 1)
    else:
        scores["Significance"] = round(avg_sim * 100, 1)

    # Fairness — low std = fair
    if len(off_diag) > 1:
        scores["Fairness"] = round(max(0, 1 - np.std(off_diag) * 3) * 100, 1)
    else:
        scores["Fairness"] = 100.0

    return scores

# ── Plotting Functions (Sunset Orange palette) ─────────────────────────────────
BG      = "#0d0d0d"
CARD    = "#1a1108"
BORDER  = "#c2410c"
ORANGE  = "#f97316"
AMBER   = "#fbbf24"
CREAM   = "#fef3c7"
MUTED   = "#92400e"

def style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=CREAM)
    ax.xaxis.label.set_color(AMBER)
    ax.yaxis.label.set_color(AMBER)
    ax.title.set_color(CREAM)

def plot_standards_radar(paul_scores):
    cats = list(paul_scores.keys())
    vals = [paul_scores[c] for c in cats]
    N = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)]
    vals_p = vals + [vals[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#1a1108")

    ax.plot(angles, vals_p, 'o-', linewidth=2.2, color=ORANGE)
    ax.fill(angles, vals_p, alpha=0.22, color=ORANGE)

    # Add value labels on each point
    for angle, val in zip(angles[:-1], vals):
        ax.annotate(f"{val:.0f}%",
                    xy=(angle, val),
                    xytext=(angle, val + 6),
                    ha='center', va='center',
                    color=AMBER, fontsize=8, fontweight='bold')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color=AMBER, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], color=MUTED, fontsize=7)
    ax.grid(color=BORDER, linestyle='--', alpha=0.4)
    ax.spines['polar'].set_color(BORDER)
    ax.set_title("Paul's Standards — Radar", color=CREAM, fontsize=12,
                 fontweight='bold', pad=18)
    plt.tight_layout()
    return fig

def plot_standards_bar(paul_scores):
    cats = list(paul_scores.keys())
    vals = [paul_scores[c] for c in cats]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    style_ax(ax, fig)

    # Color each bar based on score
    bar_colors = [ORANGE if v >= 70 else AMBER if v >= 45 else "#dc2626" for v in vals]
    bars = ax.barh(cats, vals, color=bar_colors, edgecolor='none', height=0.52)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left',
                color=CREAM, fontsize=10, fontweight='bold')

    ax.set_xlim(0, 118)
    ax.set_xlabel("Score (%)", fontsize=10)
    ax.set_title("Paul's Standards — Scores", fontsize=12, fontweight='bold', pad=10)
    for lbl in ax.get_yticklabels():
        lbl.set_color(AMBER); lbl.set_fontweight('bold')
    for lbl in ax.get_xticklabels():
        lbl.set_color(MUTED)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color=ORANGE, label='High ≥70%'),
              Patch(color=AMBER,  label='Mid ≥45%'),
              Patch(color='#dc2626', label='Low <45%')]
    ax.legend(handles=legend, loc='lower right',
              facecolor=CARD, edgecolor=BORDER,
              labelcolor=CREAM, fontsize=8)
    plt.tight_layout()
    return fig

def plot_similarity_heatmap(sim_df):
    n = len(sim_df)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4.2, n * 0.9)))
    style_ax(ax, fig)

    from matplotlib.colors import LinearSegmentedColormap
    sunset_cmap = LinearSegmentedColormap.from_list(
        "sunset", ["#1a1108", "#92400e", "#c2410c", "#f97316", "#fbbf24"])

    im = ax.imshow(sim_df.values, cmap=sunset_cmap, vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=CREAM)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=CREAM)
    cbar.set_label("Similarity", color=AMBER, fontsize=9)
    cbar.outline.set_edgecolor(BORDER)

    ax.set_xticks(range(len(sim_df.columns)))
    ax.set_yticks(range(len(sim_df.index)))
    ax.set_xticklabels(sim_df.columns, rotation=45, ha='right', color=AMBER, fontsize=9, fontweight='bold')
    ax.set_yticklabels(sim_df.index, color=AMBER, fontsize=9, fontweight='bold')

    for i in range(n):
        for j in range(n):
            val = sim_df.values[i][j]
            txt_color = "#0d0d0d" if val > 0.55 else CREAM
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=txt_color, fontsize=9, fontweight='bold')

    ax.set_title("Cosine Similarity Heatmap", color=CREAM, fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig

def plot_topk_bar(pairs, top_k):
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_k]
    labels = [f"{p[0]} vs {p[1]}" for p in pairs_sorted]
    values = [p[2] for p in pairs_sorted]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4))
    style_ax(ax, fig)

    # Gradient from amber → orange based on rank
    grad = [plt.cm.YlOrRd(0.4 + 0.5 * (1 - i / max(len(labels)-1, 1))) for i in range(len(labels))]
    bars = ax.bar(labels, values, color=grad, edgecolor='none', width=0.52,
                  linewidth=0)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f'{val:.3f}', ha='center', va='bottom',
                color=CREAM, fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Cosine Similarity", fontsize=10)
    ax.set_title(f"Top-{top_k} Most Similar Pairs", fontsize=12, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=28, ha='right', color=AMBER, fontsize=9)
    plt.yticks(color=MUTED, fontsize=8)
    plt.tight_layout()
    return fig

# ── Main Analysis ──────────────────────────────────────────────────────────────
if analyse:
    filled = [t.strip() for t in texts if t.strip()]
    if len(filled) < 2:
        st.warning("⚠️ Please enter at least 2 texts before analysing.")
    else:
        with st.spinner("🔥 Computing similarities..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(filled)
            sim_matrix = cosine_similarity(embeddings)

        labels = [f"T{i+1}" for i in range(len(filled))]
        sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
        pairs = [(labels[i], labels[j], sim_matrix[i][j])
                 for i in range(len(filled)) for j in range(i+1, len(filled))]

        st.success("✅ Analysis Complete!")
        st.markdown("---")

        # ── Similarity Matrix ──────────────────────────────────────────────────
        st.markdown("### 📊 Similarity Matrix")
        from matplotlib.colors import LinearSegmentedColormap
        st.dataframe(
            sim_df.style.format("{:.4f}").background_gradient(
                cmap="YlOrRd", vmin=0, vmax=1),
            use_container_width=True
        )

        # ── Pairwise Score Cards ───────────────────────────────────────────────
        st.markdown("### 🔢 Pairwise Similarity Scores")
        top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:8]
        pair_cols = st.columns(min(len(top_pairs), 4))
        for idx, (a, b, score) in enumerate(top_pairs):
            bar_color = "#f97316" if score > 0.7 else "#fbbf24" if score > 0.4 else "#dc2626"
            pct = int(score * 100)
            with pair_cols[idx % 4]:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="score-label">{a} vs {b}</div>
                  <div class="score-value">{score:.4f}</div>
                  <div style="background:#0d0d0d;border-radius:6px;height:6px;margin-top:6px;overflow:hidden;">
                    <div style="background:{bar_color};width:{pct}%;height:100%;border-radius:6px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Similarity Graphs ──────────────────────────────────────────────────
        st.markdown("### 📈 Similarity Visualizations")
        g1, g2 = st.columns(2)
        with g1:
            fig_heat = plot_similarity_heatmap(sim_df)
            st.pyplot(fig_heat); plt.close()
        with g2:
            fig_topk = plot_topk_bar(pairs, top_k)
            st.pyplot(fig_topk); plt.close()

        st.markdown("---")

        # ── Paul's Standards ───────────────────────────────────────────────────
        st.markdown("### 🧠 Paul's Critical Thinking Standards")
        st.caption("Scores derived from text structure, vocabulary, and semantic similarity patterns.")

        paul_scores = compute_paul_standards(filled, sim_matrix)

        standard_icons = {
            "Clarity":"🎯","Accuracy":"✅","Precision":"📐",
            "Relevance":"🔗","Logic":"🧠","Significance":"⭐","Fairness":"⚖️"
        }
        standard_desc = {
            "Clarity":     "Sentence length & punctuation clarity.",
            "Accuracy":    "Semantic coherence across all texts.",
            "Precision":   "Vocabulary richness & specificity.",
            "Relevance":   "Topical relatedness — max similarity.",
            "Logic":       "Consistency of relationships (low variance).",
            "Significance":"Meaningful distinctions between pairs.",
            "Fairness":    "Balanced similarity across all comparisons.",
        }

        # Cards row
        std_cols = st.columns(4)
        for idx, (std, score) in enumerate(paul_scores.items()):
            bar_color = "#f97316" if score >= 70 else "#fbbf24" if score >= 45 else "#dc2626"
            score_color = "#f97316" if score >= 70 else "#fbbf24" if score >= 45 else "#ef4444"
            with std_cols[idx % 4]:
                st.markdown(f"""
                <div class="paul-card">
                  <div class="score-label">{standard_icons[std]} {std}</div>
                  <div class="score-value" style="color:{score_color};">{score}%</div>
                  <div style="background:#0d0d0d;border-radius:6px;height:5px;margin:6px 0 5px 0;overflow:hidden;">
                    <div style="background:{bar_color};width:{score}%;height:100%;border-radius:6px;"></div>
                  </div>
                  <div class="score-desc">{standard_desc[std]}</div>
                </div>
                """, unsafe_allow_html=True)

        # Paul's Graphs
        st.markdown("#### 📊 Standards Graphs")
        pg1, pg2 = st.columns(2)
        with pg1:
            fig_radar = plot_standards_radar(paul_scores)
            st.pyplot(fig_radar); plt.close()
        with pg2:
            fig_sbar = plot_standards_bar(paul_scores)
            st.pyplot(fig_sbar); plt.close()

        # ── Interpretation ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📝 Interpretation")
        best_pair  = max(pairs, key=lambda x: x[2])
        worst_pair = min(pairs, key=lambda x: x[2])
        avg_all    = np.mean([p[2] for p in pairs])
        top_std    = max(paul_scores, key=paul_scores.get)
        low_std    = min(paul_scores, key=paul_scores.get)

        st.markdown(f"""
<div class="section-card">

🟠 **Most Similar Pair:** `{best_pair[0]}` & `{best_pair[1]}` — score `{best_pair[2]:.4f}` — semantically closest texts.

🔴 **Least Similar Pair:** `{worst_pair[0]}` & `{worst_pair[1]}` — score `{worst_pair[2]:.4f}` — most divergent in meaning.

📊 **Overall Average Similarity:** `{avg_all:.4f}` — {"texts share a broad common topic." if avg_all > 0.5 else "texts cover diverse or unrelated topics."}

⭐ **Strongest Paul's Standard:** `{top_std}` at **{paul_scores[top_std]}%** — best intellectual quality in your input.

⚠️ **Weakest Paul's Standard:** `{low_std}` at **{paul_scores[low_std]}%** — consider revising for improvement.

</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div style="background:#1a1108;border:1.5px dashed #92400e;border-radius:14px;
            padding:1.6rem;text-align:center;color:#d97706;font-size:1rem;">
  🔥 Enter your texts above and click <strong style="color:#f97316;">Analyse</strong> to begin.
</div>
""", unsafe_allow_html=True)
