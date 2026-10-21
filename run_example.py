import numpy as np
from bootstrap_sample_size import BootstrapSampleSize
import matplotlib.pyplot as plt


rng = np.random.default_rng(42)
data = rng.beta(a=0.5, b=0.5, size=3000)


plt.hist(data, bins=30, edgecolor="black")
plt.title("Beta(0.5, 0.5)")
plt.show()


model = BootstrapSampleSize(
    data=data,
    n_bootstrap=2000,
    n_steps=100,
    metrics=np.mean,
    margin_error=0.03,
    confidence=0.95,
    seed=42
)


n_sample = model.find_n_sample()





