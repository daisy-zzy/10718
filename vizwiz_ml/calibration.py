"""Monotone, per-label Platt scaling of frozen model logits."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import LABELS

MIN_SLOPE = 1e-6


def _logits(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS) or not np.isfinite(values).all():
        raise ValueError(f"Expected finite N x {len(LABELS)} logits")
    return values


def fit_platt(logits, targets, l2=1e-4):
    """Fit sigmoid(a*z+b), a>=MIN_SLOPE, with unweighted validation log loss.

    Small L2 regularization toward identity (a=1, b=0) prevents unconstrained
    coefficients on separable data. Both target classes must exist per label.
    This function only fits the supplied arrays; the caller enforces val-only use.
    """
    z = _logits(logits)
    y = np.asarray(targets, dtype=np.float64)
    if y.shape != z.shape or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Calibration needs nonempty binary targets matching the logits")
    if not np.isfinite(l2) or l2 <= 0:
        raise ValueError("Calibration l2 must be finite and positive")
    parameters = {}
    for j, label in enumerate(LABELS):
        x, truth = z[:, j], y[:, j]
        positives = int(truth.sum())
        if positives == 0 or positives == len(truth):
            raise ValueError(f"Cannot calibrate {label}: validation needs both positive and negative images")

        def objective(theta):
            a, b = theta
            t = a * x + b
            residual = expit(t) - truth
            penalty = 0.5 * l2 * ((a - 1) ** 2 + b ** 2)
            loss = np.mean(np.logaddexp(0, t) - truth * t) + penalty
            gradient = np.array([np.mean(residual * x) + l2 * (a - 1),
                                 np.mean(residual) + l2 * b])
            return float(loss), gradient

        result = minimize(objective, [1.0, 0.0], jac=True, method="L-BFGS-B",
                          bounds=[(MIN_SLOPE, None), (None, None)],
                          options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Platt fit failed for {label}: {result.message}")
        a, b = map(float, result.x)
        calibrated_logits = a * x + b
        parameters[label] = {
            "a": a, "b": b, "positives": positives, "negatives": len(truth) - positives,
            "validation_nll_before": float(np.mean(np.logaddexp(0, x) - truth * x)),
            "validation_nll_after": float(np.mean(np.logaddexp(0, calibrated_logits) - truth * calibrated_logits))}
    return {"format_version": 1, "method": "per_label_platt", "labels": list(LABELS),
            "l2": float(l2), "minimum_slope": MIN_SLOPE, "images": len(y),
            "parameters": parameters}


def platt_logits(logits, calibration):
    z = _logits(logits)
    if (calibration.get("format_version") != 1 or calibration.get("method") != "per_label_platt"
            or calibration.get("labels") != list(LABELS)):
        raise ValueError("Unsupported calibration format or label order")
    parameters = calibration.get("parameters", {})
    if set(parameters) != set(LABELS):
        raise ValueError("Calibration must contain all seven labels")
    a = np.array([parameters[label]["a"] for label in LABELS], dtype=np.float64)
    b = np.array([parameters[label]["b"] for label in LABELS], dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < MIN_SLOPE).any():
        raise ValueError("Calibration requires finite parameters and strictly positive slopes")
    return a * z + b


def apply_platt(logits, calibration):
    return expit(platt_logits(logits, calibration))


def check_calibration(calibration, checkpoint_sha256, protocol):
    """Refuse coefficients fitted to another model, dataset, or split."""
    if calibration.get("fit_split") != "val":
        raise ValueError("Only validation-fitted calibration is supported")
    if calibration.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Calibration was fitted to a different checkpoint")
    if calibration.get("protocol") != protocol:
        raise ValueError("Calibration dataset/protocol does not match the checkpoint")
