# AI Essay Grader

An automated essay scoring and feedback tool built on classical ML/NLP. Given an essay, it predicts a quality score and returns specific, feature-based feedback — built as a real-world alternative to manual grading for tutors and coaching institutes.

**[Live Demo](#)** *(add your Render URL here once deployed)*

## Problem

Coaching institutes and tutors can't scale manual essay grading — it's slow, inconsistent, and doesn't scale past a handful of students. This tool automates first-pass scoring and feedback, freeing up tutor time for higher-value review.

## Dataset

[ASAP (Automated Student Assessment Prize)](https://www.kaggle.com/c/asap-aes) — Essay Set 1: 1,783 real persuasive student essays, scored 2–12 by human graders.

## Approach

**Feature engineering:** 5 linguistic features extracted per essay — word count, sentence count, vocabulary richness, average sentence length, and a lightweight regex-based grammar-error heuristic (no external dependencies).

**Text representation:** TF-IDF vectorization (500 features, unigrams + bigrams) combined with the engineered features.

**Modeling:** Trained and compared three approaches on a fixed 70/15/15 train/val/test split:
- Naive baseline (predict the mean training score)
- Ridge Regression
- Random Forest Regressor (chosen model)

**Evaluation metric:** Quadratic Weighted Kappa (QWK) — the standard metric for ordinal essay scoring, since plain accuracy treats a 1-point error the same as a 10-point error.

**Feedback generation:** Percentile-based thresholds (10th/25th/75th/90th) computed from training data, mapped to specific, feature-level feedback — not generic praise/criticism.

## Results

| Model | Validation QWK |
|---|---|
| Naive baseline | 0.0000 |
| Ridge Regression | 0.8032 |
| Random Forest | 0.8364 |

**Final Test QWK: 0.7734** — at the top end of typical human-grader agreement (0.6–0.75), on data the model never saw during training or tuning.

## Known Limitations

Documenting these honestly rather than hiding them:

- **Underpredicts top-tier essays.** The model reliably scores low-to-mid essays accurately but tends to underpredict genuinely excellent (11-12/12) essays, likely due to class imbalance — very few top-scoring essays exist in the training data. A known, common failure mode in essay-scoring models.
- **Sensitive to punctuation.** Sentence segmentation uses simple punctuation-based splitting. Essays with missing or inconsistent punctuation (e.g., run-on text with few periods) will get inaccurate sentence-count and average-sentence-length features.
- **Single essay type.** Trained only on ASAP Set 1 (persuasive essays). Scores and feedback are not calibrated for other essay types or rubrics.
- **Score display note.** The 0-100 display score is a linear rescale of the underlying 2-12 model prediction, not independently learned at that resolution — it reflects relative position on the original scale, not fine-grained 100-point precision.

## Tech Stack

Python, pandas, scikit-learn, Flask, TF-IDF (scikit-learn), Quadratic Weighted Kappa evaluation

## Project Structure

```
essay-scoring-tool/
├── app.py                  # Flask API + frontend route
├── templates/
│   └── index.html          # Web UI
├── model.pkl                # trained Random Forest model
├── vectorizer.pkl           # fitted TF-IDF vectorizer
├── thresholds.pkl           # feature percentile thresholds
├── requirements.txt
├── notebooks/
│   └── Essay_reader.ipynb   # full training pipeline
└── data/                    # ASAP dataset (not tracked in git)
```

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

## API

**POST** `/score`
```json
{ "essay": "your essay text here" }
```
Returns predicted score (2-12 and 0-100 scale) and feature-level feedback.

## Future Work

- Fine-tuned transformer model, compared against this classical baseline on the same QWK metric
- Generalize to additional ASAP essay sets
- LLM-generated (rather than templated) feedback text
- Address top-score underprediction via class-weighting or targeted data augmentation