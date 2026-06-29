import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

st.set_page_config(page_title="Text Similarity App", layout="wide")

st.title("Text Similarity using Pretrained NLP Model")

st.write("This application uses the pretrained model all-MiniLM-L6-v2.")

text1 = st.text_input("Enter Sentence 1")
text2 = st.text_input("Enter Sentence 2")
text3 = st.text_input("Enter Sentence 3")

if st.button("Compare Similarity"):

    if text1 and text2 and text3:

        sentences = [text1, text2, text3]

        model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = model.encode(sentences)

        similarity = cosine_similarity(embeddings)

        st.subheader("Similarity Matrix")

        similarity_df = pd.DataFrame(
            similarity,
            index=["Sentence 1", "Sentence 2", "Sentence 3"],
            columns=["Sentence 1", "Sentence 2", "Sentence 3"]
        )

        st.dataframe(similarity_df)

        st.subheader("Similarity Scores")

        score12 = similarity[0][1]
        score13 = similarity[0][2]
        score23 = similarity[1][2]

        st.write("Sentence 1 vs Sentence 2 :", round(score12,4))
        st.write("Sentence 1 vs Sentence 3 :", round(score13,4))
        st.write("Sentence 2 vs Sentence 3 :", round(score23,4))

        st.subheader("Bar Chart")

        pair_names = [
            "S1 vs S2",
            "S1 vs S3",
            "S2 vs S3"
        ]

        scores = [
            score12,
            score13,
            score23
        ]

        fig, ax = plt.subplots()

        ax.bar(pair_names, scores)

        ax.set_ylabel("Similarity Score")

        st.pyplot(fig)

        st.subheader("Heatmap")

        fig2, ax2 = plt.subplots()

        sns.heatmap(
            similarity_df,
            annot=True,
            cmap="Blues",
            ax=ax2
        )

        st.pyplot(fig2)

        st.subheader("2D Embedding Plot")

        pca = PCA(n_components=2)

        reduced = pca.fit_transform(embeddings)

        fig3, ax3 = plt.subplots()

        ax3.scatter(
            reduced[:,0],
            reduced[:,1]
        )

        labels = [
            "Sentence 1",
            "Sentence 2",
            "Sentence 3"
        ]

        for i in range(len(labels)):
            ax3.text(
                reduced[i][0],
                reduced[i][1],
                labels[i]
            )

        st.pyplot(fig3)

        st.subheader("Paul's Critical Thinking Standards")

        st.markdown("### Clarity")
        st.write("The entered sentences are compared using semantic similarity.")

        st.markdown("### Accuracy")
        st.write("Model Used: all-MiniLM-L6-v2. No training or preprocessing has been performed.")

        st.markdown("### Precision")
        st.write("Exact similarity scores are displayed.")

        st.markdown("### Relevance")
        st.write("The graphs directly represent similarity results.")

        st.markdown("### Logic")
        highest = max(scores)

        if highest == score12:
            st.write("Sentence 1 and Sentence 2 are the most similar because they have the highest similarity score.")

        elif highest == score13:
            st.write("Sentence 1 and Sentence 3 are the most similar because they have the highest similarity score.")

        else:
            st.write("Sentence 2 and Sentence 3 are the most similar because they have the highest similarity score.")

        st.markdown("### Significance")
        st.write("The highest similarity pair is the most important result.")

        st.markdown("### Fairness")
        st.write("The pretrained model may not understand every domain-specific sentence perfectly.")

    else:

        st.warning("Please enter all three sentences.")
