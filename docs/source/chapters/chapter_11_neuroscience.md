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

## 11.2  Spike sorting in one paragraph

Modern spike sorters (Kilosort 4, MountainSort 5) factor the recorded waveform matrix as `W ≈ TC`, where `T` are spike *templates* (per neuron) and `C` are sparse *codes* (when each neuron fired). Deep variants learn templates via dictionary learning and apply graph clustering for merge / split decisions. Always validate with paired ground truth or at minimum refractory-period violation rates.

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

## 11.5  Connectomics — segmentation at petabyte scale

The FlyWire and MICrONS connectomes process ~1 PB volumes. The dominant pipeline:

1. **Affinity prediction** — 3-D UNet predicts per-voxel boundary affinities.
2. **Agglomeration** — mean / median agglomeration into supervoxels, then mergers (often by a separate GNN).
3. **Synapse detection** — second CNN predicts pre/post-synaptic partners.
4. **Proofreading** — a *queue* of human-in-the-loop edits guided by an uncertainty model.

Plan for a year of cluster time per cubic millimeter; pipelines such as `cloudvolume`, `nglui`, and `synful` are standard.

## 11.6  Pitfalls

- **Brain-score worship.** A model that matches voxel responses can still be wrong about computation. Triangulate with behavior and lesion studies.
- **Spike sorting drift.** Long recordings drift; modern sorters track templates over time, but always inspect drift maps.
- **Cross-subject decoding.** Models trained on one subject's brain do not transfer; use subject-aligned latents (CEBRA-Time, REFORM).

## 11.7  Exercises

1. **Reproduce LFADS.** Fit LFADS on the MC-Maze NLB-21 benchmark. Report co-smoothing.
2. **Brain-score.** Pick three CNNs (AlexNet, ResNet-50, CLIP). Compute their Brain-Score on V4 / IT data. Discuss the trend.
3. **Connectome neighborhood.** Pull a 50 × 50 × 50 µm³ FlyWire volume; run a 3-D UNet for affinity prediction; visualize segmentations in `neuroglancer`.
4. **Decoding under drift.** Hold out the final 30 % of a long Neuropixels session as test. Compare a static linear decoder vs. an online recalibrated one.

## 11.8  Further reading

- Pachitariu, M. *Kilosort.* (2024).
- Pandarinath, C. *LFADS.* Nat. Methods (2018).
- Schneider, S. *CEBRA.* Nature (2023).
- Dorkenwald, S. *FlyWire whole-brain connectome.* Nature (2023).

## See also

- [Chapter 6 — Modeling Living Systems](chapter_06_modeling.md)
- [Chapter 12 — Behavior & Social Systems](chapter_12_ethology.md)
- [Chapter 18 — Experimental Design](chapter_18_experiments.md)
