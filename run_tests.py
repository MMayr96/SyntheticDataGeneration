import numpy as np
import warnings
from KernelSynth import FastPhysicsEngine

# Suppress Numba compiler warnings about parallel=True
warnings.filterwarnings('ignore', category=UserWarning)

print("Initializing FastPhysicsEngine...")
engine = FastPhysicsEngine()

print("Generating batch...")
x, adj, lags, scores = engine.generate_validated(batch_size=2, n_nodes=5, length=100)
print(f"Generated shapes: {x.shape}, {adj.shape}")
print(f"Validation Scores: {scores}")

from KernelSynthDiverse import FastPhysicsEngine as FPEDiverse
engine_div = FPEDiverse()
x, adj, lags, scores = engine_div.generate_validated(batch_size=2, n_nodes=5, length=100)
print(f"Generated Diverse shapes: {x.shape}, {adj.shape}")
print(f"Validation Diverse Scores: {scores}")
print("ALL TESTS PASSED!")
