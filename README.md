# ActTune

Source implementation of core components from **The ActTune Design** in the
manuscript. This release is not yet the complete implementation used for the
paper's experiments.

The release contains Python, CUDA and C++ source, dependency configuration,
examples and tests. Calibration observations, pretrained weights, fitted banks,
fitted trees, hardware tables, logs and benchmark results are not included.

The tree and packed kernels include original source; the calibration interface,
bank integration and controller also contain newly written portable code.
Model-specific OFT/pi0.5 calibration, scale folding and optimized inference
integration have not been fully migrated. Calibration observation collection,
paired hardware measurement and the complete LIBERO closed-loop evaluation
entry points are also missing. These are code gaps, separate from the deliberate
exclusion of datasets and generated artifacts. Unit and kernel tests do not
establish reproduction of the paper's success, latency or energy results.

## Method and code

| Paper component | Implementation |
|---|---|
| 13 previous-action features plus current proprioception | `actune/features.py`, `actune/tree.py` |
| Equal suite / trajectory / context weights; loss-directed splits; depth at most 5; at least 20 contexts per child | `actune/tree.py`, `actune/_tree_growth.py` |
| Exact bottom-up subtree pruning for every feasible leaf count | `actune/_tree_growth.py` |
| Validation action loss and CPU traversal cost; explicit choice of tree size | `python -m actune fit-tree` |
| Action-Jacobian reconstruction weights, activation-guided scaling and row clipping | `actune/calibration.py` |
| W4A8/W4A4 average budgets, per-shape activation limits, two binary switchable matrices and full resident-bank storage | `actune/allocation.py`, `actune/precision.py` |
| Four configurations `q00`, `q01`, `q11`, `q10`; fallback `q11` | `actune/precision.py`, `actune/layers.py` |
| Joint frequency/power search subject to mean inference latency <= 1.10 x reference | `actune/hardware.py` |
| Bounded state forecast, asynchronous updates between calls, readback, two-request minimum dwell | `actune/runtime.py`, `actune/device/` |
| Packed W2/W4/W8 and A2/A4/A8 integer execution | `actune/kernels/` |
| Shape/precision-specific CUDA Graphs and upstream model execution helpers | `actune/graph.py`, `actune/models/` |

The online selector is a configuration-loss tree. C0--C2 K-means labels belong
to the paper's diagnostic experiments and are not inputs to this selector.
Three leaves are the paper's chosen experimental operating point; fitting does
not impose three leaves or use automatic one-standard-error selection.

## Install and check

```bash
python -m pip install -e .
python -m unittest discover -s tests -p test_method.py -v
```

For calibration and native kernels, use a CUDA-compatible PyTorch/Triton
environment and a local CUDA toolkit/C++ compiler:

```bash
python -m pip install -e '.[cuda]'
python -m unittest discover -s tests -v
```

CUDA extensions build from the included sources into PyTorch's extension cache.
No compiled libraries are committed. `TORCH_EXTENSIONS_DIR` and
`TRITON_CACHE_DIR` can point to writable local cache directories. The integer
kernels target NVIDIA Ampere or newer; the source was checked on an RTX A6000.
GPU tests are skipped when CUDA is unavailable. CPU tests can be run using only
NumPy with the first command above.

## Fit a shared precision tree

Create fitting and validation `.npz` files outside this repository. Each contains:

| Array | Shape / meaning |
|---|---|
| `x` | `[N, 13 + state_dim]`, action features in `ACTION_FORECAST_FEATURE_NAMES` order, then proprioception |
| `losses` | `[N, 4]`, postprocessed BF16 action disagreement in `q00,q01,q11,q10` order |
| `has_history` | `[N]` boolean; false at episode/trajectory starts |
| `suite` | `[N]` string suite identifiers |
| `trajectory` | `[N]` string original demonstration IDs, unique within each suite |
| `observation_id` | Optional `[N]` unique observation identities for overlap checking |

Split whole trajectories before calibrating the bank. For OFT, score each suite
with its corresponding checkpoint/bank. For pi0.5, share the checkpoint/bank.
Every configuration must use the same observation and stochastic policy noise.
Build history from the preceding fallback-policy prediction, never the current
reference prediction or future actions. Missing/nonfinite history uses `q11`.

```bash
python -m actune fit-tree --fit data/calibration/fit.npz \
  --validation data/calibration/validation.npz --output outputs/tree_candidates
```

Compare candidate validation losses and measured traversal costs across the
model/budget settings. To export the paper's three-leaf operating point:

```bash
python -m actune fit-tree --fit data/calibration/fit.npz \
  --validation data/calibration/validation.npz --leaves 3 --output outputs/frozen_tree
```

Validation never refits splits or leaf configurations. A requested leaf count
must be feasible in the grown tree. Traversal timings exclude feature extraction,
GPU inference and switching overhead.

## Quantization and model integration

`calibrate_linear` returns activation-guided channel scales, clipping ratios
and reconstruction errors for each local W/A pair. `jacobian_row_weights` takes
a differentiable BF16 action computation, including denoising for pi0.5.
`build_banks` uses those training sensitivities and explicit backend byte counts;
`select_bank` scores the three base configurations with training action loss.
Exclude the unused vocabulary projection and keep the action head,
proprioception projection, embeddings, norms and attention products unchanged.

`install_bank` installs the included group-64 backend into an already loaded
PyTorch backbone. Its calibration inputs/errors must use that same group-64
activation quantizer. It checks loaded storage against the declared byte counts.
Channel scales may be folded into adjacent modules; if supplied as runtime
input scales, their extra storage is also checked. Vision layers that must keep
one CUDA Graph assignment can be listed in `fixed_layers` during allocation.

Tokenwise A8/A4 kernels are also retained in `actune/kernels/` for existing
calibrated integrations. Do not reuse a bank calibrated for a different
activation quantizer. In particular, this portable group-64 installer is not a
drop-in loader for historical OFT experiment artifacts.

See `examples/integrate.py` for wiring a loaded model to `Controller`. The
upstream OFT/OpenPI model loaders and preprocessing remain external dependencies.
The included model helpers preserve OFT's action-head computation and pi0.5's
ten denoising steps. Supply postprocessed actions and the continuous gripper
margin using the model's own decision boundary.

## Hardware control

The reference point is 1800 MHz / 300 W. `select_hardware_policy` consumes paired
training replays of supported frequency/power points, with the precision tree
and bank frozen. It rejects action or memory mismatches and searches uniform
settings and latency-penalty tables. `verify_diagnostic_policy` checks supplied
diagnostic measurements and reverts the entire table to reference when those
checks fail. The caller must collect the measurements on disjoint trajectories;
this helper does not perform replay or enforce that separation. Unsupported
pairs use the calibrated fallback.

`Controller` waits for a pending update before inference. After the selected
policy call completes, it forecasts the next region/configuration and queues a
hardware change. A forecast mismatch leaves the prepared operating point
active; precision always uses the current observation. It performs exactly one
policy call per request. A controller owns one episode stream and one GPU;
sharing a GPU among independent controllers is unsupported.

For physical control, `PersistentDCGMClocks` requires an existing DCGM
hostengine, explicit GPU index/UUID and a group containing only that GPU.
The `dcgmi` executable must be on `PATH`, and the DCGM Python bindings must be
importable in the active environment. No installation directory is hardcoded.
It keeps the initial memory clock, verifies requested frequency and power cap,
and restores the original settings on close. The controller must be closed
before the device owner. No test changes GPU clock or power settings.

The 10% limit in the offline search is an estimate. Verify complete-call mean
latency in closed-loop evaluation separately, including residual waits. Energy
per success includes failed episodes and gaps between calls, not just GEMMs.
The included unit/kernel checks are not a reproduction of the manuscript's
full LIBERO success, speed or energy results.

## Source lineage

The loss-tree growth/pruning comes from the original calibration implementation.
Feature extraction and allocation
come from `bits_observation`; packed kernels and quantizers come from the local
VLA harness and `experiments/pi05_w4a4_shared512`. CUDA Graph and kernel tuning
sources come from the paper's optimized inference implementation.
The portable training interface, controller and bank integration connect these
components to the revised method and remove machine-specific artifact paths.
Upstream license text for the harness is retained in `licenses/`.

The pre-existing `answer.md` is retained as a historical draft. This README and
the `actune` package describe the released implementation.
