# Magma

**Momentum-Aligned Gradient Masking for Adaptive Optimizers**

Magma is a lightweight wrapper that applies block-wise stochastic masking to any PyTorch optimizer, modulated by the alignment between gradient momentum and the current gradient. It is an implementation of the algorithm described in *"On Surprising Effectiveness of Masking Updates in Adaptive Optimizers"*[(arXiv 2602.15322)](https://arxiv.org/pdf/2602.15322).

The core insight is deceptively simple. At each step, a per-parameter Bernoulli coin flip decides whether to keep or discard the update. Updates that survive are further scaled by a smoothed cosine similarity score between the gradient and its exponential moving average. The base optimizer's internal states i.e Adam's running means or RMSProp's squared gradients are always updated. Only the parameter itself is masked.

This acts as a form of implicit regularization, particularly effective under the heterogeneous curvature and heavy-tailed gradient noise characteristic of transformer training.

## Installation

```bash
pip install magma-optimizer
```

Or directly from source:

```bash
pip install git+https://github.com/andrijdavid/magma-optimizer.git
```

## Usage

Magma wraps any instantiated PyTorch optimizer. The interface mirrors what you already know.

```python
from magma import Magma
import torch

model = ...  # your model
base = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer = Magma(
    base,
    mask_prob=0.5,        # prob of keeping an update
    tau=2.0,              # temperature for the alignment sigmoid
    momentum_beta=0.9,    # EMA coefficient for gradient momentum
    alignment_ema=0.9,    # EMA coefficient for smoothing the alignment score
    exclude=set(model.embed.parameters()),  # skip masking on embeddings
)

for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

The `exclude` parameter accepts a set of tensors that should bypass masking entirely. The paper recommends excluding embedding layers, as their update dynamics differ from attention and MLP blocks.

## Algorithm

The procedure, applied at each step for each non-excluded parameter:

1. Update momentum EMA: `μ = β·μ + (1−β)·g`
2. Compute alignment: `s̃ = sigmoid(cosine_similarity(μ, g) / τ)`
3. Smooth alignment: `s = 0.9·s_prev + 0.1·s̃`
4. Run the base optimizer step (all internal states update normally)
5. Sample mask: `m ~ Bernoulli(p)`
6. Apply: `θ = (s·m)·θ_new + (1 − s·m)·θ_old`

When the mask is zero, the parameter reverts to its pre-step value. When the mask is one, the update is scaled by the alignment score. The base optimizer sees every gradient regardless.

## Citation

```bibtex
@article{joo2026magma,
  title={On Surprising Effectiveness of Masking Updates in Adaptive Optimizers},
  author={Joo, Taejong and Xia, Wenhan and Kim, Cheolmin and Zhang, Ming and Ie, Eugene},
  journal={arXiv preprint arXiv:2602.15322},
  year={2026}
}
```

## License

MIT
