import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="News Classifier | Real-Time Pattern Recognition",
    page_icon="📰",
    layout="wide"
)

# --- SIDEBAR: CONFIGURATION ---
st.sidebar.header("1. Data & Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

# Define Category Mapping (Based on your Report Figure 2)
# NOTE: Ensure these IDs match your 'category_encoded' column. 
# If the encoding was alphabetical, this list should be alphabetical.
CATEGORY_MAP = {
    0: "ARTS & CULTURE", 1: "BUSINESS", 2: "COMEDY", 3: "CRIME",
    4: "EDUCATION", 5: "ENTERTAINMENT", 6: "ENVIRONMENT", 7: "MEDIA",
    8: "POLITICS", 9: "RELIGION", 10: "SCIENCE", 11: "SPORTS",
    12: "TECH", 13: "WOMEN"
}

# --- MAIN TITLE ---
st.title("📰 Real-Time News Classification System")
st.markdown("""
*Pattern Recognition & Analysis (STTHK3013)* **Technique:** Linear Support Vector Machine (SVM) with TF-IDF Vectorization  
""")

# --- STATE MANAGEMENT ---
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'accuracy' not in st.session_state:
    st.session_state['accuracy'] = 0.0

# --- TAB LAYOUT ---
tab1, tab2, tab3 = st.tabs(["📊 Model Training", "⚡ Live Prediction", "📡 Stream Simulation"])

# ==========================================
# TAB 1: MODEL TRAINING & EVALUATION
# ==========================================
with tab1:
    st.header("Model Development Dashboard")
    
    if uploaded_file is not None:
        # Load Data
        df = pd.read_csv(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data Preview")
            st.dataframe(df.head(5))
            
        with col2:
            st.subheader("Dataset Info")
            st.info(f"Total Records: {len(df)}")
            if 'category_encoded' in df.columns:
                st.write("Target Distribution:")
                st.bar_chart(df['category_encoded'].value_counts())
        
        st.divider()

        # Training Controls
        st.subheader("Training Configuration")
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("Test Split Size", 0.1, 0.4, 0.2)
        with c2:
            st.write("Preprocessing applied automatically: `TF-IDF (max_features=5000)`")

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Preprocessing data and training Linear SVM..."):
                # --- LOGIC FROM PERSON 2'S NOTEBOOK ---
                try:
                    # 1. Feature Definition
                    X = df['clean_text'].astype(str)
                    y = df['category_encoded']

                    # 2. Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )

                    # 3. Pipeline Definition (Exactly as in Pattern.ipynb)
                    model_pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(max_features=5000)),
                        ('clf', LinearSVC(C=1.0, dual=False))
                    ])

                    # 4. Fit
                    model_pipeline.fit(X_train, y_train)
                    
                    # Store in Session State
                    st.session_state['model'] = model_pipeline
                    
                    # 5. Predictions & Metrics
                    y_pred = model_pipeline.predict(X_test)
                    
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average='weighted')
                    rec = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    st.session_state['accuracy'] = acc

                    # --- DISPLAY RESULTS ---
                    st.success("Training Complete!")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{acc:.2%}")
                    m2.metric("Precision", f"{prec:.2%}")
                    m3.metric("Recall", f"{rec:.2%}")
                    m4.metric("F1-Score", f"{f1:.2%}")
                    
                    # Confusion Matrix Plot
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error during training. Check dataset columns. Details: {e}")

    else:
        st.warning("⚠️ Please upload the `cleaned_news_dataset.csv` file in the sidebar to begin.")

# ==========================================
# TAB 2: MANUAL PREDICTION
# ==========================================
with tab2:
    st.header("Single Article Classification")
    
    if st.session_state['model']:
        user_input = st.text_area("Enter News Headline or Content:", height=150, placeholder="E.g., The stock market reached an all-time high today...")
        
        if st.button("Predict Category"):
            if user_input:
                prediction_id = st.session_state['model'].predict([user_input])[0]
                prediction_label = CATEGORY_MAP.get(prediction_id, f"Category {prediction_id}")
                
                st.subheader("Prediction Result:")
                st.markdown(f"### 🏷️ **{prediction_label}**")
                
                # Visual confidence (SVM doesn't give proba by default, so we simulate visual feedback)
                st.info(f"Model classified this using the trained LinearSVC decision boundary.")
            else:
                st.error("Please enter some text.")
    else:
        st.error("Model not trained yet. Go to the 'Model Training' tab first.")

# ==========================================
# TAB 3: REAL-TIME STREAM SIMULATION
# ==========================================
with tab3:
    st.header("🔴 Live News Stream Simulation")
    st.markdown("This module simulates a real-time feed of news coming from an API.")

    if st.session_state['model']:
        # Dummy Stream Data
        stream_data = [
            "NASA announces new mission to Mars scheduled for 2030.",
            "The Lakers secured a victory in the final seconds of the game.",
            "Global markets rally as inflation data shows improvement.",
            "New education policy aims to reduce student debt.",
            "Review: The latest superhero movie breaks box office records.",
            "Senate passes bill to improve infrastructure across the country.",
            "Researchers discover new species of deep sea fish.",
            "Local artist wins prestigious international award."
        ]

        if st.button("Start Live Stream"):
            st.write("---")
            placeholder = st.empty()
            
            for article in stream_data:
                # Prediction
                pred_id = st.session_state['model'].predict([article])[0]
                pred_label = CATEGORY_MAP.get(pred_id, f"Cat {pred_id}")
                
                # Dynamic Display
                with placeholder.container():
                    st.markdown(f"**Incoming Feed:** `{article}`")
                    
                    # Color coding based on category (Example logic)
                    if "SPORTS" in pred_label: color = "green"
                    elif "BUSINESS" in pred_label: color = "blue"
                    elif "POLITICS" in pred_label: color = "red"
                    else: color = "orange"
                    
                    st.markdown(f"Detected Category: :{color}[**{pred_label}**]")
                    st.progress(100) # Simple visual indicator
                    
                time.sleep(1.5) # Simulate delay
            
            st.success("Stream simulation finished.")
            
    else:
        st.error("Model not trained yet. Go to the 'Model Training' tab first.")

# --- FOOTER ---
st.markdown("---")
st.caption("STTHK3013 Mini Project | Developed by Person 3")