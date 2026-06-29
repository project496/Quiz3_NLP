import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SemantiQ – Text Similarity Analyzer", layout="wide")

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f0f1a; }
    h1 { color: #ffffff; font-weight: 700; }
    h2, h3 { color: #c4b5fd; }
    .stTextArea textarea { background: #1a1a2e; color: #e2e8f0; border: 1px solid #4c1d95; border-radius: 8px; }
    .stSlider > div > div { background: #4c1d95; }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.8rem; font-weight: 600; font-size: 1rem;
        transition: all 0.3s; cursor: pointer;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #a855f7, #c084fc); transform: translateY(-1px); }
    .metric-card {
        background: #1a1a2e; border: 1px solid #4c1d95;
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.4rem 0;
    }
    .score-label { color: #a78bfa; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; }
    .score-value { color: #ffffff; font-size: 1.6rem; font-weight: 700; }
    .stInfo { background: #1e1b4b; border: 1px solid #4c1d95; border-radius: 10px; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🔬 SemantiQ — Text Similarity & Critical Thinking Analyzer")
st.markdown("Enter 2 to 10 texts. The model computes **cosine similarity** between ALL pairs and evaluates them against **Paul's Intellectual Standards**.")

# ─── Sidebar Instructions ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 Instructions")
    st.info("""
**Paul's Critical Thinking Standards:**

| Standard | What It Measures |
|---|---|
| 🎯 Clarity | How unambiguous the texts are |
| ✅ Accuracy | Semantic precision of content |
| 📐 Precision | Specificity & detail level |
| 🔗 Relevance | Topical relatedness across texts |
| 🧠 Logic | Consistency in meaning |
| ⭐ Significance | Importance of differences |
| ⚖️ Fairness | Balance across all comparisons |
""")
    st.markdown("---")
    st.markdown("**Model:** `all-MiniLM-L6-v2`")
    st.markdown("**Metric:** Cosine Similarity (0–1)")

# ─── Text Input Section ────────────────────────────────────────────────────────
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
                    placeholder=f"Enter word, sentence, or paragraph #{text_idx + 1}...",
                    height=100,
                    key=f"text_{text_idx}"
                )
                texts.append(t)

# ─── Top-K Slider ─────────────────────────────────────────────────────────────
st.markdown("---")
col_k, col_btn, col_reset = st.columns([3, 1, 1])
with col_k:
    top_k = st.slider("Top-K similar results to show", min_value=1, max_value=max(1, (n_texts * (n_texts - 1)) // 2), value=3)

with col_btn:
    analyse = st.button("🚀 Analyse Similarity")

with col_reset:
    if st.button("🔄 Reset"):
        st.rerun()

# ─── Analysis ─────────────────────────────────────────────────────────────────

def compute_paul_standards(texts, sim_matrix):
    """
    Derive Paul's 7 Intellectual Standards as percentage scores
    from the actual text content and cosine similarity matrix.
    """
    scores = {}
    n = len(texts)
    
    # ── 1. CLARITY — based on average sentence length & punctuation
    # Short, well-punctuated sentences = clearer
    clarity_scores = []
    for t in texts:
        words = t.split()
        word_count = len(words)
        # Optimal clarity: 10–20 words. Penalize very short or very long.
        if word_count == 0:
            clarity_scores.append(0)
        else:
            optimal = 1 - abs(word_count - 15) / 50  # peaks at 15 words
            punctuation_bonus = 0.1 if any(c in t for c in ".!?,") else 0
            clarity_scores.append(max(0, min(1, optimal + punctuation_bonus)))
    scores["Clarity"] = round(np.mean(clarity_scores) * 100, 1)

    # ── 2. ACCURACY — based on average pairwise similarity
    # High similarity among topically consistent texts → accurate/precise use
    off_diag = []
    for i in range(n):
        for j in range(i+1, n):
            off_diag.append(sim_matrix[i][j])
    avg_sim = np.mean(off_diag)
    # Moderate similarity (0.4–0.8) is ideal; too low = unrelated, too high = redundant
    accuracy = 1 - abs(avg_sim - 0.6) / 0.6
    scores["Accuracy"] = round(max(0, min(1, accuracy)) * 100, 1)

    # ── 3. PRECISION — based on vocabulary richness (unique words ratio)
    precision_scores = []
    for t in texts:
        words = re.findall(r'\w+', t.lower())
        if not words:
            precision_scores.append(0)
        else:
            unique_ratio = len(set(words)) / len(words)
            # More unique words = more precise / specific
            length_bonus = min(len(words) / 30, 0.3)  # longer texts slightly rewarded
            precision_scores.append(min(1, unique_ratio * 0.8 + length_bonus))
    scores["Precision"] = round(np.mean(precision_scores) * 100, 1)

    # ── 4. RELEVANCE — based on max pairwise similarity
    # If at least some pairs are highly similar → relevant to each other
    if off_diag:
        max_sim = max(off_diag)
        relevance = max_sim  # 0–1 naturally
    else:
        relevance = 0
    scores["Relevance"] = round(relevance * 100, 1)

    # ── 5. LOGIC — based on consistency of similarity scores (low variance = logical)
    if len(off_diag) > 1:
        variance = np.var(off_diag)
        # Low variance = consistent semantic relationships = logical structure
        logic = max(0, 1 - variance * 10)
    else:
        logic = avg_sim
    scores["Logic"] = round(logic * 100, 1)

    # ── 6. SIGNIFICANCE — based on spread between highest and lowest similarity
    # Wide spread = some pairs clearly more significant/related than others
    if len(off_diag) > 1:
        spread = max(off_diag) - min(off_diag)
        significance = min(1, spread * 2)
    else:
        significance = avg_sim
    scores["Significance"] = round(significance * 100, 1)

    # ── 7. FAIRNESS — based on how evenly distributed the similarities are
    # Fairly balanced = each text contributes similarly
    if len(off_diag) > 1:
        std = np.std(off_diag)
        fairness = max(0, 1 - std * 3)
    else:
        fairness = 1.0
    scores["Fairness"] = round(fairness * 100, 1)

    return scores

def plot_standards_radar(standards_scores):
    """Radar / Spider chart for Paul's Standards"""
    categories = list(standards_scores.keys())
    values = [standards_scores[c] for c in categories]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values_plot = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    ax.plot(angles, values_plot, 'o-', linewidth=2, color='#a855f7')
    ax.fill(angles, values_plot, alpha=0.25, color='#7c3aed')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='#c4b5fd', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='#6b7280', fontsize=7)
    ax.grid(color='#4c1d95', linestyle='--', alpha=0.5)
    ax.spines['polar'].set_color('#4c1d95')
    ax.set_title("Paul's Standards — Overview", color='white', fontsize=13, fontweight='bold', pad=20)

    return fig

def plot_standards_bar(standards_scores):
    """Horizontal bar chart for Paul's Standards with percentage labels"""
    categories = list(standards_scores.keys())
    values = [standards_scores[c] for c in categories]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    colors = ['#7c3aed', '#a855f7', '#c084fc', '#9333ea', '#6d28d9', '#8b5cf6', '#4c1d95']
    bars = ax.barh(categories, values, color=colors, edgecolor='none', height=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left', color='white', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 115)
    ax.set_xlabel("Score (%)", color='#a78bfa', fontsize=10)
    ax.set_title("Paul's Critical Thinking Standards — Scores", color='white', fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors='#c4b5fd', labelsize=10)
    ax.spines['bottom'].set_color('#4c1d95')
    ax.spines['left'].set_color('#4c1d95')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#1a1a2e')

    for label in ax.get_yticklabels():
        label.set_color('#c4b5fd')
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        label.set_color('#9ca3af')

    plt.tight_layout()
    return fig

def plot_similarity_heatmap(sim_df):
    """Similarity heatmap"""
    fig, ax = plt.subplots(figsize=(max(5, len(sim_df) * 1.1), max(4, len(sim_df))))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    import matplotlib.colors as mcolors
    cmap = plt.cm.get_cmap('BuPu')

    im = ax.imshow(sim_df.values, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.set_label("Similarity", color='white', fontsize=10)

    ax.set_xticks(range(len(sim_df.columns)))
    ax.set_yticks(range(len(sim_df.index)))
    ax.set_xticklabels(sim_df.columns, rotation=45, ha='right', color='#c4b5fd', fontsize=9)
    ax.set_yticklabels(sim_df.index, color='#c4b5fd', fontsize=9)

    for i in range(len(sim_df)):
        for j in range(len(sim_df.columns)):
            val = sim_df.values[i][j]
            color = 'black' if val > 0.5 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9, fontweight='bold')

    ax.set_title("Pairwise Cosine Similarity Heatmap", color='white', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig

def plot_topk_bar(pairs, top_k):
    """Bar chart for Top-K similar pairs"""
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_k]
    labels = [f"{p[0]} vs {p[1]}" for p in pairs_sorted]
    values = [p[2] for p in pairs_sorted]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.3), 4))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    gradient_colors = plt.cm.plasma(np.linspace(0.3, 0.85, len(labels)))
    bars = ax.bar(labels, values, color=gradient_colors, edgecolor='none', width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Cosine Similarity", color='#a78bfa', fontsize=10)
    ax.set_title(f"Top-{top_k} Most Similar Text Pairs", color='white', fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(axis='x', colors='#c4b5fd', labelsize=9)
    ax.tick_params(axis='y', colors='#9ca3af', labelsize=9)
    ax.spines['bottom'].set_color('#4c1d95')
    ax.spines['left'].set_color('#4c1d95')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return fig

# ─── Main Logic ───────────────────────────────────────────────────────────────
if analyse:
    filled = [t.strip() for t in texts if t.strip()]
    if len(filled) < 2:
        st.warning("⚠️ Please enter at least 2 texts before analysing.")
    else:
        with st.spinner("🔄 Loading model and computing similarities..."):
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(filled)
            sim_matrix = cosine_similarity(embeddings)

        labels = [f"T{i+1}" for i in range(len(filled))]
        sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)

        # All pairs
        pairs = []
        for i in range(len(filled)):
            for j in range(i+1, len(filled)):
                pairs.append((labels[i], labels[j], sim_matrix[i][j]))

        st.success("✅ Analysis Complete!")
        st.markdown("---")

        # ── Similarity Matrix Table ──────────────────────────────────────────
        st.markdown("### 📊 Similarity Matrix")
        st.dataframe(sim_df.style.format("{:.4f}").background_gradient(cmap='BuPu'), use_container_width=True)

        # ── Quick Score Cards ────────────────────────────────────────────────
        st.markdown("### 🔢 Pairwise Scores")
        pair_cols = st.columns(min(len(pairs), 4))
        for idx, (a, b, score) in enumerate(sorted(pairs, key=lambda x: x[2], reverse=True)[:8]):
            with pair_cols[idx % len(pair_cols)]:
                color = "#22c55e" if score > 0.7 else "#f59e0b" if score > 0.4 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="score-label">{a} vs {b}</div>
                    <div class="score-value" style="color:{color};">{score:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Graphs Row 1: Heatmap + Top-K Bar ───────────────────────────────
        st.markdown("### 📈 Similarity Visualizations")
        g1, g2 = st.columns(2)
        with g1:
            fig_heat = plot_similarity_heatmap(sim_df)
            st.pyplot(fig_heat)
            plt.close()
        with g2:
            fig_topk = plot_topk_bar(pairs, top_k)
            st.pyplot(fig_topk)
            plt.close()

        st.markdown("---")

        # ── Paul's Standards ─────────────────────────────────────────────────
        st.markdown("### 🧠 Paul's Critical Thinking Standards Analysis")
        st.caption("Scores computed from text properties (length, vocabulary richness, punctuation) and semantic similarity structure.")

        paul_scores = compute_paul_standards(filled, sim_matrix)

        # Score metric cards
        standard_icons = {
            "Clarity": "🎯", "Accuracy": "✅", "Precision": "📐",
            "Relevance": "🔗", "Logic": "🧠", "Significance": "⭐", "Fairness": "⚖️"
        }
        standard_desc = {
            "Clarity":     "How clear and unambiguous your texts are (sentence length & punctuation).",
            "Accuracy":    "Semantic coherence — how well texts stick to a consistent topic.",
            "Precision":   "Vocabulary richness & specificity (unique word ratio).",
            "Relevance":   "Topical relatedness — highest pairwise similarity score.",
            "Logic":       "Consistency of relationships across all pairs (low variance = logical).",
            "Significance":"Spread between highest and lowest similarity — meaningful distinctions.",
            "Fairness":    "Balance of similarity distribution — no single pair dominates.",
        }

        std_cols = st.columns(4)
        for idx, (standard, score) in enumerate(paul_scores.items()):
            color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
            with std_cols[idx % 4]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="score-label">{standard_icons[standard]} {standard}</div>
                    <div class="score-value" style="color:{color};">{score}%</div>
                    <div style="color:#6b7280; font-size:0.75rem; margin-top:0.3rem;">{standard_desc[standard]}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("#### 📊 Paul's Standards — Graphs")
        pg1, pg2 = st.columns(2)
        with pg1:
            fig_radar = plot_standards_radar(paul_scores)
            st.pyplot(fig_radar)
            plt.close()
        with pg2:
            fig_bar = plot_standards_bar(paul_scores)
            st.pyplot(fig_bar)
            plt.close()

        # ── Textual Interpretation ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📝 Interpretation")
        best_pair = max(pairs, key=lambda x: x[2])
        worst_pair = min(pairs, key=lambda x: x[2])
        overall_avg = np.mean([p[2] for p in pairs])

        st.markdown(f"""
- **Most Similar Pair:** `{best_pair[0]}` & `{best_pair[1]}` — similarity score `{best_pair[2]:.4f}` → these texts are semantically closest.
- **Least Similar Pair:** `{worst_pair[0]}` & `{worst_pair[1]}` — similarity score `{worst_pair[2]:.4f}` → these texts differ the most in meaning.
- **Overall Average Similarity:** `{overall_avg:.4f}` — {"texts are broadly related." if overall_avg > 0.5 else "texts cover diverse or unrelated topics."}
- **Paul's Top Standard:** `{max(paul_scores, key=paul_scores.get)}` scored **{max(paul_scores.values())}%** — strongest intellectual quality in your input.
- **Paul's Weakest Standard:** `{min(paul_scores, key=paul_scores.get)}` scored **{min(paul_scores.values())}%** — consider revising for improvement.
""")
else:
    st.info("👆 Enter your texts above and click **Analyse Similarity** to begin.")
