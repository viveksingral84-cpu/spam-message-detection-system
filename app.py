"""
Spam Message Detection System - Streamlit Web Application
=========================================================
A clean, interactive, and beginner-friendly web interface to classify
text messages as 'Spam' or 'Not Spam' (Ham) in real-time.
"""

import os
import json
import streamlit as st
import joblib

# Set page configuration
st.set_page_config(
    page_title="Spam Message Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .result-card-spam {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 1.2rem;
        color: #991B1B;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-card-ham {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 1.2rem;
        color: #065F46;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-spam {
        background-color: #EF4444;
        color: white;
    }
    .badge-ham {
        background-color: #10B981;
        color: white;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")


@st.cache_resource
def load_model_and_vectorizer():
    """Loads the trained MultinomialNB model and fitted TF-IDF vectorizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None


@st.cache_data
def load_metrics():
    """Loads model evaluation metrics from json."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def predict_message(message: str, model, vectorizer):
    """
    Transforms input text with TF-IDF vectorizer and predicts class & probabilities.
    Returns: label (str), confidence (float), prob_ham (float), prob_spam (float), top_features (list)
    """
    # Transform message using fitted vectorizer
    transformed_text = vectorizer.transform([message])

    # Predict class (0: Ham, 1: Spam)
    prediction = model.predict(transformed_text)[0]
    probabilities = model.predict_proba(transformed_text)[0]

    prob_ham = probabilities[0]
    prob_spam = probabilities[1]

    if prediction == 1:
        label = "Spam"
        confidence = prob_spam
    else:
        label = "Not Spam"
        confidence = prob_ham

    # Find matching keywords present in the model's vocabulary
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = transformed_text.nonzero()[1]
    detected_keywords = [feature_names[idx] for idx in nonzero_indices]

    return label, confidence, prob_ham, prob_spam, detected_keywords


# Initialize session state for input text
if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""


def set_sample_text(sample_text: str):
    """Helper to fill sample messages into input box."""
    st.session_state["user_text"] = sample_text


def clear_text():
    """Resets the input text area."""
    st.session_state["user_text"] = ""


def main():
    # Load model artifacts
    model, vectorizer = load_model_and_vectorizer()
    metrics = load_metrics()

    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
        st.title("About Project")
        st.markdown("""
        **Spam Message Detection System** is an AI/ML application that classifies incoming SMS/text messages as legitimate or fraudulent.
        
        - **Algorithm**: Multinomial Naive Bayes
        - **Feature Extractor**: TF-IDF Vectorizer
        - **Dataset**: UCI SMS Spam Collection (5,572 messages)
        """)

        st.divider()

        # Model Performance Section
        st.subheader("Model Performance")
        if metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Accuracy", value=f"{metrics.get('accuracy', 0)*100:.2f}%")
                st.metric(label="Precision", value=f"{metrics.get('precision', 0)*100:.2f}%")
            with col2:
                st.metric(label="Recall", value=f"{metrics.get('recall', 0)*100:.2f}%")
                st.metric(label="F1-Score", value=f"{metrics.get('f1_score', 0)*100:.2f}%")
            st.caption(f"Evaluated on {metrics.get('test_samples', 0)} unseen test messages (80/20 train-test split).")
        else:
            st.info("Train the model using `python train_model.py` to view metrics.")

        st.divider()

        # Quick Test Samples
        st.subheader("Quick Test Samples")
        st.markdown("Click any sample below to test instantly:")

        if st.button("Sample 1: Lottery Spam", use_container_width=True):
            set_sample_text("Congratulations! You have won a $1,000 Walmart Gift Card. Click http://claim-prize.xyz now to claim your cash award! Free entry.")

        if st.button("Sample 2: Urgent Banking Spam", use_container_width=True):
            set_sample_text("URGENT: Your bank account has been suspended due to suspicious activity. Call 0800-123-456 immediately to verify your identity.")

        if st.button("Sample 3: Legitimate Friend Chat", use_container_width=True):
            set_sample_text("Hey, are we still meeting for coffee at 4 PM near the library? Let me know if you are free.")

        if st.button("Sample 4: Normal Office Message", use_container_width=True):
            set_sample_text("Hi team, please find attached the meeting minutes and project timeline. Review before tomorrow's standup.")

    # --- MAIN CONTENT ---
    st.markdown('<div class="main-title">🛡️ Spam Message Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter any SMS or text message below to analyze whether it is authentic or a spam attempt.</div>', unsafe_allow_html=True)

    if model is None or vectorizer is None:
        st.error("Model files not found! Please run the training script first.")
        st.code("python train_model.py", language="bash")
        st.stop()

    # Input Box
    user_input = st.text_area(
        label="Message to analyze:",
        value=st.session_state["user_text"],
        placeholder="Type or paste a message here (e.g., 'Free entry to win cash prizes!')...",
        height=140,
        key="input_box"
    )

    # Sync back to session_state if typed manually
    st.session_state["user_text"] = user_input

    # Action Buttons: Predict and Clear
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🗑️ Clear", on_click=clear_text, use_container_width=True)

    # Prediction Logic
    if predict_clicked:
        clean_text = user_input.strip()

        if not clean_text:
            st.warning("Please enter a message before clicking Predict.")
        else:
            with st.spinner("Analyzing message with Naive Bayes model..."):
                label, confidence, prob_ham, prob_spam, detected_keywords = predict_message(clean_text, model, vectorizer)

            st.write("")
            # Display Prediction Card
            if label == "Spam":
                st.markdown(f"""
                <div class="result-card-spam">
                    <span class="metric-badge badge-spam">SPAM DETECTED</span>
                    <h2 style="margin: 0.5rem 0 0.2rem 0; color: #991B1B;">🚨 This message is likely SPAM</h2>
                    <p style="margin: 0; font-size: 0.95rem;">Model Confidence: <strong>{confidence * 100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-ham">
                    <span class="metric-badge badge-ham">LEGITIMATE MESSAGE</span>
                    <h2 style="margin: 0.5rem 0 0.2rem 0; color: #065F46;">✅ This message is NOT SPAM (Ham)</h2>
                    <p style="margin: 0; font-size: 0.95rem;">Model Confidence: <strong>{confidence * 100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

            # Confidence Breakdown
            st.subheader("Confidence Score Breakdown")
            col_h, col_s = st.columns(2)
            with col_h:
                st.write(f"**Not Spam (Ham) Probability:** `{prob_ham * 100:.2f}%`")
                st.progress(float(prob_ham))
            with col_s:
                st.write(f"**Spam Probability:** `{prob_spam * 100:.2f}%`")
                st.progress(float(prob_spam))

            # Vocabulary terms detected in input
            if detected_keywords:
                with st.expander("Detected Key Terms Analyzed by TF-IDF"):
                    st.write("The model identified the following terms from its learned training vocabulary in your message:")
                    st.write(", ".join([f"`{kw}`" for kw in detected_keywords]))

    st.write("")
    st.divider()

    # --- EDUCATIONAL SECTION ---
    with st.expander("ℹ️ How Does This Prediction Work? (Machine Learning Explanation)", expanded=False):
        st.markdown("""
        ### Step-by-Step Prediction Workflow

        1. **Text Preprocessing**:
           - When you submit a message, it is converted to lowercase and English stop words (like *'the'*, *'is'*, *'at'*) are filtered out so that only meaningful content words remain.

        2. **TF-IDF Vectorization (Term Frequency - Inverse Document Frequency)**:
           - Machine learning models cannot understand raw English text directly; they require numerical feature vectors.
           - **Term Frequency (TF)** measures how frequently a word appears in this specific message.
           - **Inverse Document Frequency (IDF)** diminishes the weight of words that appear universally across all messages and increases the importance of rare, distinguishing words (such as *"prize"*, *"urgent"*, *"guaranteed"*, *"claim"*).

        3. **Multinomial Naive Bayes Classifier**:
           - The model applies **Bayes' Theorem** with the *naive* conditional independence assumption:
        """)
        st.latex(r"P(\text{Spam} \mid \text{Words}) = \frac{P(\text{Spam}) \times \prod_{i=1}^{n} P(w_i \mid \text{Spam})}{P(\text{Words})}")
        st.markdown("""
           - It compares the probability of the message being **Spam** versus **Not Spam (Ham)**.
           - Laplace smoothing ($\alpha = 0.1$) is used to prevent zero-probability errors for words not seen in training.
           - Whichever class receives the higher posterior probability is chosen as the final prediction, and the normalized probability provides the **confidence score**.
        """)


if __name__ == "__main__":
    main()

