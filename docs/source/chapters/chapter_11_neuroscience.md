# Chapter 11 — Neuroscience & Cognitive Biology

> *"To model the brain we need models that can be brain-shaped: high-dimensional, recurrent, and noisy."*

## Learning objectives

- Survey the dominant ML problems in neuroscience: spike sorting, neural decoding, latent-dynamics inference, connectome reconstruction.
- Use latent variable models (LFADS, CEBRA) to extract task-relevant dynamics from population recordings.
- Apply convolutional and transformer models to EM connectomics segmentation.
- Critically discuss "brain-score" benchmarks and their limits.

## 11.1  The instruments

| Technique | Spatial | Temporal | Channels | AI workload |
|-----------|---------|----------|----------|-------------|
| Patch clamp | 1 neuron | µs | 1 | Minimal |
| Two-photon Ca²⁺ | 1 µm | 50 ms | 1 000–10 000 | Image processing, deconvolution |
| Neuropixels | 20 µm | 30 µs | 384–1 000 | Spike sorting (Kilosort 4) |
| fMRI | mm | s | 100 000 voxels | Decoding, RSA |
| EEG / MEG | cm | ms | 64–306 | Source localization |
| EM connectomics | nm | static | ~10¹⁰ voxels | Segmentation, synapse detection |

### 11.1a  The instruments — a practical guide to neural data modalities

The table above lists instruments but does not say *which* to reach for given a scientific question. The decision matrix below maps common questions to the best-fit modality and its trade-offs.

| Question | Best method | Spatial resolution | Temporal resolution | Scale (neurons) | Invasiveness | Cost |
|----------|-------------|--------------------|---------------------|-----------------|--------------|------|
| Which neurons fire together? | Calcium imaging (2-photon, miniscope) | Single-cell (~1 µm) | ~100 ms (indirect) | 100–10,000 | Low (cranial window) | High |
| What is the precise spike timing? | Neuropixels / silicon probes | Single-unit (50 µm) | µs | 100–1,000 (simultaneous) | Moderate | Moderate–high |
| How do many brain regions coordinate? | fMRI (BOLD) | 1–3 mm (voxel) | 1–2 s | Whole brain (millions of voxels) | Non-invasive | Very high |
| What is the local field potential (LFP) rhythm? | LFP from depth electrodes | ~500 µm (source) | <1 ms | Local population (thousands) | Moderate | Low–moderate |
| Which neurons connect to which? (connectomics) | EM reconstruction (volume EM) | Synaptic (nm) | Snapshot | Complete circuit (mm³ scale) | Terminal (post-mortem) | Very high (petascale) |
| How do neurons encode behavior? | Combined electrophysiology + videography | Single-unit + pose | µs (spikes) + video (30 Hz) | 100–1,000 + pose | Moderate | High |
| What is the causal role of a neuron type? | Optogenetics + recording | Targeted (opsin-expressing) | ms (light pulses) | Variable (up to thousands) | High (viral injection + implant) | Moderate–high |

**Rule of thumb:** Start with wide-field calcium imaging (least invasive, high throughput) for discovery. Follow up with Neuropixels for high-precision spike timing. Use fMRI only for human studies or large-scale primate mapping.

**Pitfall:** Calcium imaging measures *indirect* spikes (via GCaMP fluorescence) and misses fast bursts. Always validate with patch-clamp or Neuropixels in a subset of experiments.

## 11.2  Spike sorting in one paragraph

Modern spike sorters (Kilosort 4, MountainSort 5) factor the recorded waveform matrix as `W ≈ TC`, where `T` are spike *templates* (per neuron) and `C` are sparse *codes* (when each neuron fired). Deep variants learn templates via dictionary learning and apply graph clustering for merge / split decisions. Always validate with paired ground truth or at minimum refractory-period violation rates.

### 11.2a  Spike sorting — a deeper look at Kilosort 4

A practical workflow for running Kilosort 4 on Neuropixels data, including the quality metrics that separate a usable single unit from noise.

**Key steps in Kilosort:**

1. **Preprocessing** — high-pass filtering (300 Hz), common median referencing.
2. **Spike detection** — threshold crossing (4–6 × noise standard deviation).
3. **Template matching** — iterative refinement of spike templates using a convolutional dictionary.
4. **Clustering** — graph-based clustering of templates (using drift tracking for long recordings).
5. **Merging / splitting** — based on autocorrelogram refractory-period violations and template similarity.

**Minimum quality metrics for a "good" unit:**

- **Refractory-period violation rate** (<0.5% of spikes within 1 ms).
- **Signal-to-noise ratio (SNR)** > 5 (peak-to-peak amplitude / noise floor).
- **Isolation distance** (Mahalanobis distance from other units) > 20.
- **Presence in recording** — at least 500 spikes (for stable estimates).

```python
# Using the `kilosort` wrapper (simplified)
import kilosort
import numpy as np

# Assumes binary raw data file (`.bin` or `.dat`)
ops = kilosort.default_ops()
ops['fs'] = 30000          # sampling rate (Hz)
ops['n_channels'] = 384    # Neuropixels 1.0
ops['data_dir'] = '/path/to/neuropixels_recording'

# Run kilosort
kilosort.run(ops)

# Load results (spike times, cluster IDs, templates)
spike_times = np.load(ops['data_dir'] + '/spike_times.npy')
spike_clusters = np.load(ops['data_dir'] + '/spike_clusters.npy')
templates = np.load(ops['data_dir'] + '/templates.npy')
```

**Quality assessment with the `phy` GUI** (post-processing): after Kilosort, use the `phy` template GUI to manually curate units. Look for:

- A clear refractory period in the autocorrelogram (dip at 0–1 ms).
- No multi-unit contamination (uniform waveform shape across spikes).
- Stable amplitude over time (no drift-induced splitting).

**Pitfall:** Kilosort assumes a stationary recording (no electrode drift). For long recordings (>1 hour), use Kilosort 4's drift tracking or post-hoc motion correction (e.g., `ironclust`).

## 11.3  Latent dynamics

Models like LFADS, AutoLFADS, NDT, and CEBRA assume

```
x_t  ≈  f(z_t),         z_{t+1}  =  g(z_t) + noise
```

with `z` low-dimensional. Practical recipe:

1. Bin spikes at 5–20 ms.
2. Fit a sequence VAE / contrastive model.
3. Visualize `z_t` colored by task variables; verify task-relevant geometry (e.g. rotational dynamics in motor cortex).
4. Decode behavior from `z` with a linear regressor — non-linearity rarely helps beyond a good latent space.

### 11.3a  Latent dynamics — extending LFADS with AutoLFADS

LFADS (Latent Factor Analysis via Dynamical Systems) requires a human to pick the latent dimensionality and dynamics hyperparameters. **AutoLFADS** automates this with cross-validation, giving a principled and reproducible choice.

**AutoLFADS workflow:**

1. Bin spike counts (5–20 ms bins) for all recorded units.
2. Split data into training, validation, and test trials.
3. For each candidate latent dimension (e.g., 4, 8, 12, 16) and recurrent hidden size (32, 64, 128):
   - Train LFADS (sequence VAE with GRU dynamics).
   - Compute validation loss (negative log-likelihood of held-out spikes).
4. Choose the model with lowest validation loss.
5. Re-train on the full training set; decode latent trajectories.

**Why AutoLFADS matters:** manual selection of latent dimensionality is subjective and often leads to overfitting. AutoLFADS provides a principled, reproducible choice.

```python
# Pseudocode using a simplified LFADS implementation
import numpy as np
import lfads

# Input: spikes (n_trials, n_neurons, n_timebins)
train_data, valid_data, test_data = lfads.utils.split_trials(spike_counts)

best_val_loss = np.inf
best_hparams = None
for latent_dim in [4, 8, 12, 16]:
    for rnn_dim in [32, 64, 128]:
        model = lfads.LFADS(latent_dim=latent_dim, rnn_dim=rnn_dim,
                            input_dim=n_neurons, output_dim=n_neurons)
        model.train(train_data, valid_data, epochs=50)
        val_loss = model.evaluate(valid_data)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hparams = {'latent_dim': latent_dim, 'rnn_dim': rnn_dim}
print(f"Best hyperparameters: {best_hparams}")

# Retrain with best_hparams on the full training set
final_model = lfads.LFADS(**best_hparams)
final_model.train(train_data, epochs=100)
latent_trajectories = final_model.encode(test_data)  # (n_trials, time, latent_dim)
```

**Interpretation:** plot the latent trajectories (PCA or UMAP) colored by a task variable (e.g., hand velocity, stimulus identity). In motor cortex you should see rotational dynamics (circular trajectories) for reaching movements.

**Pitfall:** LFADS (and AutoLFADS) requires many trials (≥50) with consistent structure. For single-trial or low-trial-count experiments, use simpler methods (GPFA, PPC).

## 11.4  Worked example — CEBRA on motor cortex

```python
import cebra

model = cebra.CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    learning_rate=3e-4,
    temperature=1.0,
    output_dimension=8,
    max_iterations=10_000,
)
model.fit(spike_counts, hand_position)         # supervised by behavior
z = model.transform(spike_counts)              # (T, 8)
```

The resulting `z` decodes hand position with R² > 0.9 on most motor-cortex datasets and beats PCA + linear by 10–20 points.

### 11.4a  Worked example extension — CEBRA for hyperscanning (multi-brain)

The example above uses single-subject spike counts and behavior. **Hyperscanning** extends this to simultaneous recording from two animals (or human dyads) performing a joint task.

**Goal:** learn a joint embedding space where neural activity from two brains is aligned when they are in similar behavioral states or when they are coordinating.

**CEBRA-hyperscanning (conceptual):** use a contrastive loss that pulls together:

- Neural data from brain A at time `t` (conditioned on behavior).
- Neural data from brain B at the same time `t` (if they are engaged in a joint task).
- Time-lagged pairs (e.g., brain A at `t`, brain B at `t+Δt`) to capture leader–follower dynamics.

```python
import numpy as np
import cebra

# Assume we have:
# spikes_A: (T, n_neurons_A)
# spikes_B: (T, n_neurons_B)
# behavior: (T, d) joint behavior (e.g., relative distance, social touch)

# Train CEBRA separately on each brain (behavior as supervision), then align.
model_A = cebra.CEBRA(model_architecture='offset10-model', output_dimension=16)
model_A.fit(spikes_A, behavior)
emb_A = model_A.transform(spikes_A)

model_B = cebra.CEBRA(model_architecture='offset10-model', output_dimension=16)
model_B.fit(spikes_B, behavior)
emb_B = model_B.transform(spikes_B)

# Compute cross-brain alignment (Procrustes) after centering
from scipy.linalg import orthogonal_procrustes
R, _ = orthogonal_procrustes(emb_A.T, emb_B.T)
emb_B_aligned = (R @ emb_B.T).T
# Now compare trajectories
alignment_score = np.mean(np.corrcoef(emb_A.flatten(), emb_B_aligned.flatten()))
```

**For true hyperscanning with CEBRA:** use a `MultiCEBRA`-style extension (not yet in the standard library) that accepts two input modalities and a contrastive objective maximizing mutual information between simultaneous samples.

**Pitfall:** hyperscanning datasets are small (typically <1 hour, <5 dyads). Overfitting is a major risk. Use very low latent dimensions (4–8) and strong regularization.

## 11.5  Connectomics — segmentation at petabyte scale

The FlyWire and MICrONS connectomes process ~1 PB volumes. The dominant pipeline:

1. **Affinity prediction** — 3-D UNet predicts per-voxel boundary affinities.
2. **Agglomeration** — mean / median agglomeration into supervoxels, then mergers (often by a separate GNN).
3. **Synapse detection** — second CNN predicts pre/post-synaptic partners.
4. **Proofreading** — a *queue* of human-in-the-loop edits guided by an uncertainty model.

Plan for a year of cluster time per cubic millimeter; pipelines such as `cloudvolume`, `nglui`, and `synful` are standard.

### 11.5a  Connectomics — segmentation at petabyte scale: a practical guide to CloudVolume

A hands-on guide for accessing and processing a large EM volume using `cloudvolume` and `neuroglancer`.

**Access a public dataset (e.g., the FlyWire hemibrain):**

```python
import numpy as np
import cloudvolume as cv
import neuroglancer

# FlyWire hemibrain v1.2 (~20 TB dataset)
volume = cv.CloudVolume('precomputed://gs://flywire_v1_2/segmentation')
# Download a small subvolume (256³ voxels) around a synapse of interest
x, y, z = 10000, 15000, 2000   # coordinates in voxel space
size = 256
subvol = volume[x:x+size, y:y+size, z:z+size]  # (size, size, size) array of segment IDs

# Visualize in neuroglancer (interactive)
viewer = neuroglancer.Viewer()
with viewer.txn() as s:
    s.layers['segmentation'] = neuroglancer.SegmentationLayer(source=volume.get_neuroglancer_link())
    s.layers['image'] = neuroglancer.ImageLayer(source=volume.get_image_layer_link())
print(viewer)
```

**Processing a subvolume with a simple 3D U-Net for affinity prediction:**

```python
import torch
import torch.nn as nn

class Simple3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):  # 3 affinities: x, y, z
        super().__init__()
        self.enc1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(e3) + e2
        d1 = self.dec1(d2) + e1
        return self.out(d1)

# Load subvolume (ensure normalization to [0, 1])
input_tensor = torch.tensor(subvol.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,256,256,256)
model = Simple3DUNet()
affinities = model(input_tensor)  # (1,3,256,256,256)
```

**Pitfall:** training a 3D U-Net from scratch requires a massive labeled dataset (manually proofread segments). For most users, download pre-trained models (e.g., from MICrONS) and fine-tune on a small subvolume.

## 11.6  Pitfalls

- **Brain-score worship.** A model that matches voxel responses can still be wrong about computation. Triangulate with behavior and lesion studies.
- **Spike sorting drift.** Long recordings drift; modern sorters track templates over time, but always inspect drift maps.
- **Cross-subject decoding.** Models trained on one subject's brain do not transfer; use subject-aligned latents (CEBRA-Time, REFORM).

### 11.6a  Extended pitfalls — brain-score worship and cross-subject decoding

The pitfalls above are qualitative. Below are *quantitative sanity checks* for each.

**Brain-score sanity check:** compute the correlation between a model's brain-score (e.g., for V4) and its performance on a non-visual task (e.g., ImageNet classification). If the correlation is near 1, the brain-score may simply reflect task difficulty, not biological plausibility. Also compare brain-score across model families (CNNs vs. vision transformers) — the differences are often small (<0.05) and within noise.

**Cross-subject decoding check:** train a decoder on subject 1's neural data (e.g., to predict hand position). Test on subject 2. If performance drops by >50% (R² from 0.8 to 0.3), your model is subject-specific. To improve, use:

- **Hyperalignment** — Procrustes alignment of neural manifolds across subjects.
- **Shared response model (SRM)** — factorizes data into shared and subject-specific components.
- **CEBRA with subject as a nuisance variable** — train a contrastive model that aligns across subjects.

```python
def cross_subject_decoding(train_spikes, train_behavior, test_spikes, test_behavior):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    model.fit(train_spikes, train_behavior)
    r2_train = model.score(train_spikes, train_behavior)
    r2_test = model.score(test_spikes, test_behavior)
    drop = (r2_train - r2_test) / r2_train if r2_train > 0 else 1.0
    print(f"Train R²: {r2_train:.3f}, Test R²: {r2_test:.3f}, Drop: {drop:.1%}")
    return drop
```

**Pitfall:** cross-subject decoding is inherently limited by anatomical variability. Even with optimal alignment, some subjects may simply have different tuning curves (e.g., orientation preference). Report within-subject ceiling performance.

## 11.7  Exercises

1. **Reproduce LFADS.** Fit LFADS on the MC-Maze NLB-21 benchmark. Report co-smoothing.
2. **Brain-score.** Pick three CNNs (AlexNet, ResNet-50, CLIP). Compute their Brain-Score on V4 / IT data. Discuss the trend.
3. **Connectome neighborhood.** Pull a 50 × 50 × 50 µm³ FlyWire volume; run a 3-D UNet for affinity prediction; visualize segmentations in `neuroglancer`.
4. **Decoding under drift.** Hold out the final 30 % of a long Neuropixels session as test. Compare a static linear decoder vs. an online recalibrated one.
5. **Spike sorting comparison.** Download a publicly available Neuropixels dataset (e.g., from the Allen Brain Observatory). Run Kilosort 4 and MountainSort 5 on the same raw data. Compare the number of sorted units, refractory-period violations, and SNR. Which gives more high-quality units?
6. **AutoLFADS on a reaching dataset.** Use the MC-Maze dataset (from the Neural Latents Benchmark). Run AutoLFADS to find the optimal latent dimension. Then decode hand velocity from the latent trajectories using a linear decoder. Compare to a baseline that decodes directly from binned spikes (no latent model). Does LFADS improve decoding?
7. **CEBRA for cross-species alignment.** Take neural recordings from mice and humans performing similar tasks (e.g., visual detection). Train CEBRA separately on each species using behavior as supervision. Then align embeddings using Procrustes. Compute the correlation of trial-averaged trajectories. Are motor-cortex dynamics similar across species?
8. **Connectome graph analysis.** Download a small subgraph of the FlyWire hemibrain connectome (e.g., all neurons in the mushroom body). Compute graph statistics: degree distribution, clustering coefficient, path length. Compare to a random graph with the same number of nodes and edges. Is the connectome small-world?

## 11.8  Further reading

- Pachitariu, M. *Kilosort.* (2024).
- Pandarinath, C. *LFADS.* Nat. Methods (2018).
- Schneider, S. *CEBRA.* Nature (2023).
- Dorkenwald, S. *FlyWire whole-brain connectome.* Nature (2023).
- Steinmetz, N. A. et al. (2021). "Neuropixels 2.0: A miniaturized high-density probe for stable, long-term brain recordings." *Science* — hardware description and data examples.
- Hurwitz, C. et al. (2021). "Targeted neural dynamical modeling via deep learning." *NeurIPS* — AutoLFADS and beyond.
- Schneider, S. et al. (2024). "CEBRA: A framework for contrastive learning of neural and behavioral data." *Nature* — includes multi-animal experiments.
- Dorkenwald, S. et al. (2024). "FlyWire: Online community-based whole-brain connectomics." *Nature* — details of the production pipeline.

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 12 — Behavior & Social Systems](chapter_12_ethology.md)
- [Chapter 18 — Experimental Design](chapter_18_experiments.md)
