import pytest
import numpy as np
from app import model, prediction


def test_model_exists():
    """Test that model is created"""
    assert model is not None


def test_model_trained():
    """Test that model is trained and has coefficients"""
    assert model.coef_ is not None
    assert len(model.coef_) > 0


def test_prediction_exists():
    """Test that prediction is made"""
    assert prediction is not None


def test_prediction_value():
    """Test that prediction is a reasonable value"""
    assert prediction[0] > 0
    assert isinstance(prediction[0], (int, float, np.floating))


def test_prediction_with_new_input():
    """Test model can make predictions on new inputs"""
    new_input = np.array([[10]])
    new_prediction = model.predict(new_input)
    assert new_prediction is not None
    assert len(new_prediction) == 1
