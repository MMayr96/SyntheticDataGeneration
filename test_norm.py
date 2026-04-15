import numpy as np
from KernelSynth import FastPhysicsEngine

# Verify FastPhysicsEngine returns normalized pool signals effectively
# By mocking the subgenerators to output huge unnormalized signals, we check if cont_norm scaled them down.
engine = FastPhysicsEngine()
engine.gen_kernel.generate_batch = lambda n, l, normalize: np.ones((n, l)) * 1000.0 + np.random.randn(n, l) * 500.0
engine.gen_stat.generate_batch = lambda n, l: np.ones((n, l)) * -2000.0 + np.random.randn(n, l) * 100.0

# Prevent controls to purely test physics pool
engine.gen_control.generate_batch = lambda n, l: np.zeros((0, l))

# Override the mixer to return x unmodified
engine.mixer.apply_batch = lambda x, n_roots_list: (x, np.zeros((x.shape[0], x.shape[1], x.shape[1])), np.zeros((x.shape[0], x.shape[1], x.shape[1])))

x, _, _ = engine.generate(batch_size=1, n_nodes=2, length=100)
# We expect the output means to be roughly 0 and std around 1 because of our new pre-normalization block.
# Even if they weren't exactly 1 due to our specific simple mixer mock, they definitely won't be 1000 or -2000.
means = x.mean(axis=2)
stds = x.std(axis=2)

print("Means:", means)
print("Stds:", stds)

assert np.all(np.abs(means) < 0.1), "Means are not near zero!"
assert np.all(np.abs(stds - 1.0) < 0.1), "Stds are not near one!"
print("Normalization check passed.")
