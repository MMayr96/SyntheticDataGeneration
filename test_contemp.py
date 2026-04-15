import numpy as np
from KernelSynth import VectorizedMixerCausal

mixer = VectorizedMixerCausal()
# Shape: (B, N, L)
x_batch = np.random.randn(1, 5, 100)
roots = np.array([2])
out_x, out_adj, out_lags = mixer.apply_batch(x_batch, roots, method_probs=(0, 1, 0, 0, 0))

print("Contemp adjacency roots to physics:")
print(out_adj[0, :2, 2:])
print("Differences in physics nodes (out - original_physics_nodes) (Should not be zero matrix where roots are connected)")
# The physics nodes should have changed because they mixed the root signal
print(out_x[0, 2:, :] - x_batch[0, 2:, :])
