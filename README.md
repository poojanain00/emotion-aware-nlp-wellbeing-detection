# emotion-aware-nlp-wellbeing-detection
Code and models for an emotion-aware, explainable NLP pipeline using multi-source student feedback to detect wellbeing and engagement risks. Includes data preprocessing, machine learning, deep learning, DistilBERT fine-tuning, GPT-LoRA rationales, and XAI (SHAP/LIME).


This notebook contains the complete end-to-end implementation of the dissertation project:

“Emotion-Aware GPT Fine-Tuning and Explainable Deep Learning on Multi-Source Educational Feedback for Early Detection of Student Wellbeing and Engagement Risks.”

It integrates every stage of the research workflow in a single, reproducible pipeline, including:

1. Data Loading & Unified Schema Construction

Imports the three Zenodo datasets (RateMyProfessor, Waterloo course evaluations, Exeter reviews).

Aligns and harmonises metadata into a unified 26-field schema.

Ensures reproducibility through documented cleaning, validation checks, and dataset summaries.

2. Text Cleaning & Advanced Preprocessing

Implements the custom functions strong_preclean and advanced_clean.

Performs:

Unicode normalisation

Stopword removal

POS-aware lemmatisation

Negation-tagging (neg_word)

Token and length calculations

Produces both raw and cleaned text fields.

3. Feature Engineering

Creates structural and emotional features including:

clean_len

neg_ratio

punctuation-based intensity cues

dataset/ entity-type one-hot encodings

Maps emotion labels (Neg, Conf/Neu, Pos) to numeric classes.

4. Classical Machine Learning Models

Includes fully tuned implementations of:

Logistic Regression (TF-IDF + engineered features)

Linear SVM with probability calibration

Each model outputs confusion matrices, macro-F1 scores, ROC-AUC and feature-importance insights.

5. Deep Learning Models

Implements the full modelling pipeline for:

Artificial Neural Network with SVD-compressed vectors

BiLSTM sequence model

CNN text classifier

All trained with class weights, early stopping and validation monitoring.

6. Transformer-Based Model

Fine-tuned DistilBERT using Hugging Face Trainer API with:

tokenisation

train/validation/test splits

macro-F1 evaluation

cross-context (domain-shift) experiments

This forms the strongest model in the pipeline.

7. GPT-2 LoRA Fine-Tuning

Integrates PEFT (LoRA) for:

lightweight GPT-2 adaptation

natural-language rationale generation

perplexity reporting

qualitative explanation of wellbeing signals

Used for interpretability, not classification.

8. Explainable AI (XAI)

The pipeline includes all final explainability outputs:

SHAP token-level heatmaps for DistilBERT

LIME explanations for ML/DL models

Logistic regression coefficients

GPT-generated rationales for human-readable insights

These support transparent, responsible educational AI.

9. Final Evaluation & Results Export

Macro-F1 and weighted-F1 for all models

ROC curves

Cross-dataset comparative results
