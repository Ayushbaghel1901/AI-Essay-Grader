import os
import re
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# Load exported models and thresholds
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("thresholds.pkl", "rb") as f:
        thresholds = pickle.load(f)
except Exception as e:
    raise FileNotFoundError(f"Failed to load pickled assets from root directory: {e}")

def extract_features_single(text):
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    w_count = len(words)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    s_count = max(1, len(sentences))
    vocab_richness = len(set(words)) / w_count if w_count > 0 else 0.0
    avg_sentence_length = w_count / s_count if s_count > 0 else 0.0
    
    # Lightweight grammar heuristic
    errors = sum(1 for s in sentences if s and not s[0].isupper())
    errors += len(re.findall(r"\b(\w+)\s+\1\b", text.lower()))
    
    return {
        "word_count": w_count,
        "sentence_count": s_count,
        "vocab_richness": vocab_richness,
        "avg_sentence_length": avg_sentence_length,
        "grammar_errors": errors,
    }

def generate_feedback(features_dict):
    feedback = []
    features = ["word_count", "sentence_count", "vocab_richness", "avg_sentence_length", "grammar_errors"]
    for feat in features:
        val = features_dict.get(feat, 0)
        p10, p25, p75, p90 = thresholds[feat]
        
        if feat == "grammar_errors":
            if val <= p10:
                bucket = "bottom 10%"
                classification = "good"
            elif val <= p25:
                bucket = "10-25%"
                classification = "good"
            elif val <= p75:
                bucket = "middle range"
                classification = "good"
            elif val <= p90:
                bucket = "75-90%"
                classification = "mild notes"
            else:
                bucket = "top 10%"
                classification = "notable"
        elif feat == "vocab_richness":
            if val <= p10:
                bucket = "bottom 10%"
                classification = "notable"
            elif val <= p25:
                bucket = "10-25%"
                classification = "mild notes"
            elif val <= p75:
                bucket = "middle range"
                classification = "good"
            elif val <= p90:
                bucket = "75-90%"
                classification = "good"
            else:
                bucket = "top 10%"
                classification = "good"
        else:
            if val <= p10:
                bucket = "bottom 10%"
                classification = "notable"
            elif val <= p25:
                bucket = "10-25%"
                classification = "mild notes"
            elif val <= p75:
                bucket = "middle range"
                classification = "good"
            elif val <= p90:
                bucket = "75-90%"
                classification = "mild notes"
            else:
                bucket = "top 10%"
                classification = "notable"
            
        feedback.append(f"{feat}: value={val:.2f} ({bucket}) -> {classification}")
    return feedback

@app.route("/score", methods=["POST"])
def score():
    data = request.get_json(silent=True) or {}
    essay_text = data.get("essay", "")
    if not essay_text or not isinstance(essay_text, str):
        return jsonify({"error": "Missing or empty 'essay' field in JSON request body"}), 400
    
    # 1. Feature Extraction
    feat_dict = extract_features_single(essay_text)
    feat_df = pd.DataFrame([feat_dict])
    
    # 2. TF-IDF Transformation
    tfidf_sparse = vectorizer.transform([essay_text])
    tfidf_df = pd.DataFrame(tfidf_sparse.toarray(), index=feat_df.index)
    
    # 3. Concatenate and Align Column names
    X = pd.concat([feat_df, tfidf_df], axis=1)
    X.columns = X.columns.astype(str)
    
    # 4. Score Prediction
    raw_score = model.predict(X.values)[0]
    predicted_score = int(np.clip(np.round(raw_score), 2, 12))
    score_out_of_100 = int(np.clip(np.round((raw_score - 2) / 10 * 100), 0, 100))
    
    # 5. Feedback Generation
    feedback_list = generate_feedback(feat_dict)
    
    return jsonify({
        "predicted_score": predicted_score,
        "raw_score": float(raw_score),
        "score_out_of_100": score_out_of_100,
        "feedback": feedback_list
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)