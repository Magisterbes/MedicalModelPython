"""Quick check that all imports work after removing logistic_regression.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model.parameters import Parameters, _get_age_group, _get_age_group_vector
from model.simulation import Simulation
print("All imports OK")
print("LogisticRegression from sklearn:", type(Simulation))