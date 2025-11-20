"""
Model serialization functions for GBRS models.

This module provides functions to save and load GBRS models in JSON format,
enabling cross-language compatibility between Python and R implementations.
"""

import json
import numpy as np


def save_model(model, filepath, objective=None, formula=None):
    """
    Save a GBRS model to a JSON file.
    
    Parameters
    ----------
    model : gbrs.core.Model
        The fitted C++ model object
    filepath : str
        Path to save the model file
    objective : str, optional
        Model objective ('continuous', 'binary', or 'survival')
    formula : str, optional
        Model formula string
    """
    # Get model parameters
    params = model.get_params()
    idxs = model.get_idxs()
    split_vals = model.get_split_val()
    
    # Build rules list
    rules = []
    for i in range(len(idxs)):
        if int(idxs[i]) >= 0:  # Valid rule
            rules.append({
                "idx": int(idxs[i]),
                "split_val": float(split_vals[i]),
                "w": float(params.w[i]),
                "cst": float(params.y0) if i == 0 else 0.0
            })
    
    # Create model dict
    model_dict = {
        "version": "1.0",
        "objective": objective if objective else "unknown",
        "formula": formula if formula else "unknown",
        "rules": rules
    }
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(model_dict, f, indent=2)


def load_model(filepath):
    """
    Load a GBRS model from a JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the model file
        
    Returns
    -------
    dict
        Dictionary containing:
        - version: str
        - objective: str
        - formula: str
        - rules: list of dict with keys (idx, split_val, w, cst)
    """
    with open(filepath, 'r') as f:
        model_dict = json.load(f)
    
    # Validate version
    if model_dict.get("version") != "1.0":
        raise ValueError(f"Unsupported model version: {model_dict.get('version')}")
    
    return model_dict


def save_predictions(predictions, filepath):
    """
    Save model predictions to a JSON file for cross-language comparison.
    
    Parameters
    ----------
    predictions : array-like
        Model predictions
    filepath : str
        Path to save predictions
    """
    # Convert to list for JSON serialization
    pred_list = [float(p) for p in predictions]
    
    with open(filepath, 'w') as f:
        json.dump({"predictions": pred_list}, f, indent=2)


def load_predictions(filepath):
    """
    Load predictions from a JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to predictions file
        
    Returns
    -------
    numpy.ndarray
        Array of predictions
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return np.array(data["predictions"])
