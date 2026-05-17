import numpy as np

def explain_prediction(prediction):
    """
    Simple explainability layer for EEG BCI model
    (portfolio-friendly version)
    """

    explanations = {
        "LEFT HAND": {
            "reason": "Higher activation detected in right motor cortex (simulated CSP pattern).",
            "brain_region": "Right Motor Cortex",
            "signal_pattern": "Asymmetrical alpha suppression"
        },
        "RIGHT HAND": {
            "reason": "Higher activation detected in left motor cortex (simulated CSP pattern).",
            "brain_region": "Left Motor Cortex",
            "signal_pattern": "Contralateral motor activation"
        }
    }

    return explanations.get(prediction, {
        "reason": "No explanation available",
        "brain_region": "Unknown",
        "signal_pattern": "Unknown"
    })