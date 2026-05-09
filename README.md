# Malaria Cell-Smear AI Diagnostic Prototype

An academic AI-in-healthcare project that classifies thin-smear microscopy cell images as
`Parasitized` or `Uninfected` and explains the prediction with Grad-CAM.

This repository is intentionally framed as a responsible AI prototype, not a production
medical device. The goal is to show how prediction, confidence, interpretability, and
limitations should be presented together instead of treating accuracy as the whole story.

## What Changed In This XAI Version

- Added Grad-CAM overlays to the Streamlit app so reviewers can see which image regions most influenced the prediction.
- Added a collapsed activation-map debug view for inspecting selected MobileNetV2 feature channels.
- Fixed the app's model path so it resolves relative to `app.py`, not the terminal's current directory.
- Rewrote the README to match the actual malaria microscopy task.
- Added a lightweight model card and requirements file for reproducibility.

## Dataset

The notebook uses the public Malaria Cell Images Dataset from NIH/Kaggle.

- Task: binary image classification
- Classes: `Parasitized`, `Uninfected`
- Approximate size: 27,558 cell images
- Split used in the notebook: 80% train, 10% validation, 10% test

Important limitation: the dataset contains cropped single-cell images. It does not represent
a complete clinical microscopy workflow, slide-level diagnosis, patient-level aggregation, or
deployment in variable hospital/lab imaging conditions.

## Model

- Base model: MobileNetV2 pretrained on ImageNet
- Head: GlobalAveragePooling2D, Dropout, Dense sigmoid output
- Input size: 224 x 224 RGB
- Output: sigmoid score where `0 = Parasitized` and `1 = Uninfected`

The current model is saved at:

```text
malaria_App/malaria_cell_parasite_prediction_model.h5
```

## Explainability

The Streamlit app now generates a Grad-CAM heatmap for the predicted class and overlays it on
the uploaded image.

Interpretation guidance:

- Bright regions contributed more strongly to the model's current prediction.
- Grad-CAM shows model attention, not medical causality.
- A plausible-looking heatmap does not prove the diagnosis is correct.
- Low-confidence predictions or unclear heatmaps should be treated as cases requiring expert review.

The app also includes an activation-map debug view. This is useful for technical inspection,
but it should not be presented as clinician-facing explanation.

## Threshold Rationale and Evaluation

The dataset ZIP was fetched from the NIH/NLM public malaria cell image source and cached locally
under `data/raw/`. The raw data folder is ignored by Git because it is large, but the evaluation
script can recreate it.

Run:

```bash
python scripts/evaluate_threshold.py
```

Generated evaluation artifacts are saved under `reports/evaluation/`:

- [Threshold rationale report](reports/evaluation/threshold_rationale.md)
- [Validation threshold sweep](reports/evaluation/threshold_sweep.png)
- [Selected-threshold confusion matrix](reports/evaluation/confusion_matrix_selected_threshold.png)
- [Default-threshold confusion matrix](reports/evaluation/confusion_matrix_default_threshold.png)
- [ROC curve](reports/evaluation/roc_curve.png)
- [Metrics summary CSV](reports/evaluation/metrics_summary.csv)

Current selected threshold: `0.285` for the `Parasitized` positive class.

| Split | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1 | False Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.285 | 0.932 | 0.961 | 0.903 | 0.908 | 0.934 | 54 |
| Test | 0.285 | 0.942 | 0.960 | 0.925 | 0.924 | 0.942 | 54 |

The threshold was selected on validation data to maximize sensitivity for parasitized cells
while keeping specificity at or above `0.90`, then evaluated separately on the test split.

## Run Locally

From the repository root:

```bash
pip install -r requirements.txt
streamlit run malaria_App/app.py
```

Then upload a `.jpg`, `.jpeg`, or `.png` thin-smear cell image.

## Academic Evaluation To Add Next

The repository now includes threshold rationale, confusion matrices, and ROC analysis. The
strongest next academic improvements would be:

- False-negative and false-positive examples
- Calibration curve and Brier score
- Grad-CAM examples for true positives, true negatives, false positives, and false negatives
- PR-AUC and confidence intervals for key metrics

## Responsible Use Statement

This project is for education and research demonstration only. It is not medical advice, not a
clinical diagnostic system, and not intended for patient care. Any real diagnosis must be made
by qualified healthcare professionals using validated clinical workflows.

If deploying through a temporary public tunnel, do not upload patient-identifiable or sensitive
medical data.
