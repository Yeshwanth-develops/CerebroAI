"""
Evaluation Metrics & Patient-Level Aggregation Utilities for Medical Diagnosis.
Implemented with pure NumPy and fallback handling for Windows Application Control environments.
"""

from typing import Dict, List, Tuple, Union
import numpy as np


def compute_clinical_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred_probs: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes comprehensive clinical performance metrics using robust pure NumPy:
    - Sensitivity (Recall)
    - Specificity (True Negative Rate)
    - Balanced Accuracy
    - ROC-AUC Score
    - Precision & F1-Score
    """
    y_t = np.array(y_true, dtype=int)
    y_p_probs = np.array(y_pred_probs, dtype=float)
    y_p = (y_p_probs >= threshold).astype(int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    acc = (tp + tn) / len(y_t) if len(y_t) > 0 else 0.0
    bal_acc = (sensitivity + specificity) / 2.0

    # Fast Pure-NumPy ROC-AUC calculation (Mann-Whitney U statistic)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    if n_pos > 0 and n_neg > 0:
        ranks = np.argsort(np.argsort(y_p_probs)) + 1
        pos_ranks_sum = np.sum(ranks[y_t == 1])
        auc = float((pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    else:
        auc = 0.5

    return {
        "Accuracy": float(acc),
        "Balanced_Accuracy": float(bal_acc),
        "Sensitivity_Recall": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1_Score": float(f1),
        "ROC_AUC": float(auc),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def aggregate_patient_predictions(
    slice_records: List[Dict[str, Union[str, float, int]]],
    aggregation_method: str = "majority_vote",
    threshold: float = 0.5,
) -> List[Dict[str, Union[str, int, float, bool]]]:
    """
    Aggregates slice-level 2D predictions to compute a robust patient-level 3D diagnosis
    without requiring external C-extensions.
    """
    subjects: Dict[str, Dict[str, List]] = {}
    for r in slice_records:
        sub_id = str(r["subject_id"])
        if sub_id not in subjects:
            subjects[sub_id] = {"probs": [], "true_labels": []}
        subjects[sub_id]["probs"].append(float(r["pred_prob"]))
        subjects[sub_id]["true_labels"].append(int(r["true_label"]))

    patient_results = []
    for sub_id, data in subjects.items():
        probs = np.array(data["probs"])
        true_label = int(data["true_labels"][0])

        if aggregation_method == "mean_probability":
            patient_prob = float(np.mean(probs))
            patient_pred = int(patient_prob >= threshold)
        elif aggregation_method == "majority_vote":
            votes = (probs >= threshold).astype(int)
            patient_prob = float(np.mean(votes))
            patient_pred = int(np.sum(votes) > (len(votes) / 2))
        elif aggregation_method == "top_k_slices":
            k = max(1, int(len(probs) * 0.20))
            top_k = np.sort(probs)[-k:]
            patient_prob = float(np.mean(top_k))
            patient_pred = int(patient_prob >= threshold)
        else:
            patient_prob = float(np.mean(probs))
            patient_pred = int(patient_prob >= threshold)

        patient_results.append({
            "subject_id": sub_id,
            "true_label": true_label,
            "patient_pred_prob": patient_prob,
            "patient_prediction": patient_pred,
            "num_slices": len(probs),
            "is_correct": patient_pred == true_label,
        })

    return patient_results
