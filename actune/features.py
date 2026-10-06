"""Causal 13-dimensional action features used by ActTune."""
import numpy as np

ACTION_FORECAST_FEATURE_NAMES = (
    "translation_energy",
    "rotation_energy",
    "translation_variation",
    "rotation_variation",
    "translation_curvature",
    "rotation_curvature",
    "translation_cancellation",
    "rotation_cancellation",
    "chunk_variation",
    "chunk_curvature",
    "cancellation",
    "gripper_transition",
    "gripper_margin",
)


def _component_motion_features(component: np.ndarray) -> dict[str, float]:
    differences = np.diff(component, axis=0)
    second_differences = np.diff(component, n=2, axis=0)
    magnitude_sum = float(np.linalg.norm(component, axis=1).sum())
    cancellation = (
        0.0
        if magnitude_sum <= 1e-12
        else 1.0 - float(np.linalg.norm(component.sum(axis=0))) / magnitude_sum
    )
    return {
        "energy": float(np.linalg.norm(component, axis=1).mean()),
        "variation": (
            0.0
            if len(differences) == 0
            else float(np.linalg.norm(differences, axis=1).mean())
        ),
        "curvature": (
            0.0
            if len(second_differences) == 0
            else float(np.linalg.norm(second_differences, axis=1).mean())
        ),
        "cancellation": float(np.clip(cancellation, 0.0, 1.0)),
    }


def action_forecast_features(
    actions: np.ndarray,
    *,
    gripper_margin: float,
) -> dict[str, float]:
    """Describe the planned motion structure of one action chunk.

    Only the selected action chunk and its deterministic gripper margin are
    consumed.  The features therefore exist after chunk ``t`` and can be used
    causally for the precision decision at chunk ``t+1`` without a shadow
    forward or simulator-only state.
    """

    chunk = np.asarray(actions, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < 1:
        raise ValueError(f"expected a nonempty [H,7] action chunk, got {chunk.shape}")
    if not np.all(np.isfinite(chunk)) or not np.isfinite(gripper_margin):
        raise ValueError("action chunk and gripper margin must be finite")
    motion = chunk[:, :6]
    translation = _component_motion_features(chunk[:, :3])
    rotation = _component_motion_features(chunk[:, 3:6])
    differences = np.diff(motion, axis=0)
    second_differences = np.diff(motion, n=2, axis=0)
    magnitude_sum = float(np.linalg.norm(motion, axis=1).sum())
    cancellation = (
        0.0
        if magnitude_sum <= 1e-12
        else 1.0 - float(np.linalg.norm(motion.sum(axis=0))) / magnitude_sum
    )
    signs = np.sign(chunk[:, 6])
    return {
        "translation_energy": float(np.linalg.norm(chunk[:, :3], axis=1).mean()),
        "rotation_energy": float(np.linalg.norm(chunk[:, 3:6], axis=1).mean()),
        "translation_variation": translation["variation"],
        "rotation_variation": rotation["variation"],
        "translation_curvature": translation["curvature"],
        "rotation_curvature": rotation["curvature"],
        "translation_cancellation": translation["cancellation"],
        "rotation_cancellation": rotation["cancellation"],
        "chunk_variation": (
            0.0 if len(differences) == 0 else float(np.linalg.norm(differences, axis=1).mean())
        ),
        "chunk_curvature": (
            0.0
            if len(second_differences) == 0
            else float(np.linalg.norm(second_differences, axis=1).mean())
        ),
        "cancellation": float(np.clip(cancellation, 0.0, 1.0)),
        "gripper_transition": float(np.any(signs[1:] != signs[:-1])),
        "gripper_margin": float(gripper_margin),
    }
