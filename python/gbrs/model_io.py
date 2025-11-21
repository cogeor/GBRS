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
    model : gbrs.GBRS or gbrs.core.Model
        The fitted model object
    filepath : str
        Path to save the model file
    objective : str, optional
        Model objective ('continuous', 'binary', or 'survival')
    formula : str, optional
        Model formula string
    """
    # Handle both GBRS wrapper and raw Model object
    if hasattr(model, "_model"):
        core_model = model._model
    else:
        core_model = model
        
    # Get model parameters
    params = core_model.get_params()
    idxs = core_model.get_idxs()
    split_vals = core_model.get_split_val()
    
    # Build rules list
    rules = []
    for i in range(len(idxs)):
        # Filter out invalid rules if any (though idxs usually valid in trained model)
        # But we might have unused slots if max_n > actual rules?
        # The C++ implementation seems to fill sequentially up to 'i'.
        # But get_idxs returns the full vector of size max_n?
        # Let's check C++ get_idxs. It returns 'this->idxs'.
        # 'this->idxs' is initialized to Zero(max_n).
        # 'this->i' tracks the number of rules.
        # We should probably only save up to 'i'.
        # But we don't have access to 'i' from python directly via get_params?
        # Wait, get_params returns ScoreParams which has 'w'.
        # We can iterate and stop when we hit zeros? No, 0 is valid index.
        # Actually, the C++ prune() method resizes the vectors to 'i'.
        # So if prune() was called (which fit() does), then idxs size should be correct.
        # Let's assume vectors are correct size.
        
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
    gbrs.GBRS
        Loaded GBRS model instance
    """
    with open(filepath, 'r') as f:
        model_dict = json.load(f)
    
    # Validate version
    if model_dict.get("version") != "1.0":
        raise ValueError(f"Unsupported model version: {model_dict.get('version')}")
    
    # Create GBRS instance
    # We don't know the original hyperparameters (n_iter etc), but they don't matter for inference.
    # We can set defaults.
    from gbrs.utils import GBRS
    model = GBRS()
    
    # Set state
    model._set_state(model_dict)
    
    # Store objective/formula if we want to add them to GBRS class later
    # model.objective = model_dict.get("objective")
    # model.formula = model_dict.get("formula")
    
    return model


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
