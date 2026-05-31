# Chapter 12 — Behavior, Ethology & Social Systems

> *"Behavior is the highest-bandwidth biological measurement we have."*

## Learning objectives

- Use modern pose-estimation tools (DeepLabCut, SLEAP, Anipose) to extract markerless animal pose from video.
- Cluster behavioral time series into "syllables" with unsupervised methods (MoSeq, B-SOiD, Keypoint-MoSeq).
- Apply graph models to social network data (interactions over time) and detect emergent collective behavior.
- Recognize ethical considerations specific to non-human animal observation.

## 12.1  From pixels to pose

```mermaid
flowchart LR
  video --> det[Object detection<br/>YOLOv8]
  det --> pose[Pose estimation<br/>DeepLabCut / SLEAP]
  pose --> track[Multi-animal tracking<br/>idtracker.ai / SLEAP]
  track --> beh[Behavioral syllables<br/>MoSeq / B-SOiD]
  beh --> net[Social network<br/>networkX / pyTorch-geom]
```

Each step has a dominant tool today. Inserting a foundation pose model (e.g. ViTPose-NHP for non-human primates) typically halves the labeling burden.

### 12.1a  From pixels to pose — a decision matrix for pose-estimation tools

The pipeline above lists DeepLabCut, SLEAP, and Anipose. The table below compares the most common options to help you choose the right tool for a given experiment.

| Tool | Multi-animal? | 3D? | Labeling effort | Training time (GPU) | Occlusion handling | Best for |
|------|---------------|------|------------------|----------------------|--------------------|----------|
| **DeepLabCut** (DLC) | No (single animal) | Yes (multi-view) | Moderate (150–200 frames) | Fast (<1 hr) | Poor | Simple arenas, single animal |
| **SLEAP** | Yes (identity tracking) | No (2D) | Higher (requires identity labels) | Moderate (2–4 hr) | Good (with identity priors) | Multi-animal social interactions (2D) |
| **Anipose** | No | Yes (≥2 cameras) | Moderate (DLC models per view) | Fast per view | Fair | 3D kinematics, single animal |
| **ViTPose-NHP** | Yes (non-human primate) | No | Low (few shots) | Fast (inference only) | Good (foundation model) | Non-human primates, transfer learning |
| **DeepPoseKit** | No | No | Low (skeleton-free) | Fast | Fair | Quick prototyping, low precision |

**Rule of thumb:** Start with DeepLabCut for single-animal, 2D experiments. Switch to SLEAP for two or more animals in the same arena. Use Anipose only if you need true 3D limb kinematics (e.g. reaching). For non-human primates, try ViTPose-NHP first — it often requires <50 labeled frames.

**Pitfall:** Multi-animal tracking fails when animals overlap extensively (e.g. mating, fighting). In those frames, manual post-correction is unavoidable. Use SLEAP's "human in the loop" refinement.

## 12.2  Behavioral syllables

`Keypoint-MoSeq` (Wiltschko, Pereira et al.) fits a Gaussian-AR-HMM to keypoint time series and discovers ~30–80 distinct stereotyped motifs (syllables) per species. Each syllable has:

- A typical duration (~200–400 ms).
- A characteristic trajectory in pose space.
- A transition probability into other syllables.

These syllables provide a *vocabulary* for higher-level behavioral analyses (effect of drugs, optogenetic stimulation, genotype).

### 12.2a  Behavioral syllables — extending Keypoint-MoSeq with parameter tuning

Keypoint-MoSeq exposes a handful of hyperparameters that strongly affect the syllables it discovers. The table below is a practical tuning guide.

| Parameter | Typical range | Effect | Tuning heuristic |
|-----------|---------------|--------|-------------------|
| `num_syllables` (K) | 20–80 | Number of discrete movement motifs | Start with K=50; use elbow of log-likelihood |
| `self_transition_prior` | 10–100 | Stickiness of syllables | Higher = longer syllables; start at 50 |
| `autoregressive_order` (AR) | 2–4 | Temporal smoothness | 2 for fast movements, 4 for smooth (e.g. grooming) |
| `emission_dim` | 2–10 | Dimensionality of pose features | Use 80% variance from PCA of keypoint trajectories |
| `kappa` (concentration) | 10–100 | Peakiness of syllable emission | Lower = broader syllable definitions; start at 30 |

```python
# Pseudocode for Keypoint-MoSeq (simplified)
from moseq import extract, train, cluster

# Step 1: Extract keypoint trajectories from DLC output
keypoints = extract.load_keypoints('dlc_results.h5')

# Step 2: Train AR-HMM with grid search over K
results = []
for K in [30, 40, 50, 60, 70]:
    model = train.ARHMM(keypoints, num_states=K, ar_order=3)
    loglik = model.score(keypoints)
    results.append({'K': K, 'loglik': loglik})
# Plot loglik vs. K; choose elbow

# Step 3: Fit final model with chosen K
final_model = train.ARHMM(keypoints, num_states=chosen_K, ar_order=3)
syllables = final_model.predict(keypoints)  # array of state per frame

# Step 4: Visualize syllable usage
import matplotlib.pyplot as plt
plt.hist(syllables, bins=chosen_K)
plt.xlabel('Syllable index')
plt.ylabel('Frame count')
```

**Interpreting syllables:** Plot the mean pose trajectory for each syllable (e.g. average keypoint positions over the first 10 frames). For mouse open-field, you should see:

- Short, ballistic syllables (running)
- Curved syllables (turning)
- Stationary syllables (rearing, grooming)
- Periodic syllables (sniffing)

**Pitfall:** Syllables are not directly comparable across individuals or sessions unless you align them via hierarchical clustering or transfer learning. Use `moseq`'s cross-session alignment tool.

## 12.3  Worked example — pose → embedding → cluster

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# pose: (T, K, 2) — T frames, K keypoints, x/y
def behavior_embedding(pose: np.ndarray) -> np.ndarray:
    feats = []
    for t in range(2, len(pose)):
        v = pose[t] - pose[t - 1]          # velocity
        a = v - (pose[t - 1] - pose[t - 2]) # acceleration
        feats.append(np.concatenate([pose[t].flatten(), v.flatten(), a.flatten()]))
    X = StandardScaler().fit_transform(np.array(feats))
    return umap.UMAP(n_neighbors=30, min_dist=0.1).fit_transform(PCA(20).fit_transform(X))
```

Cluster `behavior_embedding(pose)` with HDBSCAN; play exemplar video clips per cluster to interpret.

### 12.3a  Worked example extension — pose → embedding → cluster with time-aware features

The embedding above uses velocity and acceleration. We can capture longer-range dynamics by adding **temporal window features** — the frame-to-frame displacement over a multi-frame window.

```python
import numpy as np
from scipy.signal import savgol_filter

def behavior_embedding_temporal(pose, window=5, polyorder=2):
    """
    pose: (T, K, 2) keypoint positions.
    Returns features: (T, K*2*4) with position, velocity, acceleration,
    plus windowed displacement.
    """
    T, K, _ = pose.shape
    # Smooth positions to reduce noise
    pose_smooth = np.zeros_like(pose)
    for k in range(K):
        for d in range(2):
            pose_smooth[:, k, d] = savgol_filter(pose[:, k, d], window, polyorder)

    # Velocity (1-step difference)
    vel = np.diff(pose_smooth, axis=0, prepend=pose_smooth[0:1])
    # Acceleration (1-step difference of velocity)
    acc = np.diff(vel, axis=0, prepend=vel[0:1])

    # Windowed displacement: distance between pose[t] and pose[t+window]
    disp = np.zeros((T, K, 2))
    for t in range(T - window):
        disp[t] = pose_smooth[t+window] - pose_smooth[t]
    # Last window frames get last displacement
    for t in range(T - window, T):
        disp[t] = disp[T - window - 1]

    # Concatenate features per frame
    feats = np.concatenate([
        pose_smooth.reshape(T, -1),
        vel.reshape(T, -1),
        acc.reshape(T, -1),
        disp.reshape(T, -1)
    ], axis=1)
    return feats
```

Dimension reduction and clustering:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN

feats = behavior_embedding_temporal(pose, window=10)
scaler = StandardScaler()
feats_scaled = scaler.fit_transform(feats)
pca = PCA(n_components=50)
feats_pca = pca.fit_transform(feats_scaled)
# Use HDBSCAN for density-based clustering (handles noise)
clusterer = HDBSCAN(min_cluster_size=30, min_samples=5, metric='euclidean')
labels = clusterer.fit_predict(feats_pca)
# label -1 = unassigned (noise)
```

**Visualization:** Plot UMAP of `feats_pca` colored by cluster label. Then extract video clips of each cluster and verify they correspond to interpretable behaviors (e.g. cluster 3 = rearing, cluster 5 = grooming).

**Pitfall:** Clustering in feature space yields *frame-level* labels, not *syllables*. Use a hidden Markov model (HMM) to impose temporal continuity.

## 12.4  Social-network analysis

Treat each animal as a node, each proximity / interaction event as an edge with a timestamp. Useful primitives:

- **Centrality** (degree, eigenvector, betweenness) for individuals.
- **Modularity** to detect groups.
- **Temporal motifs** (e.g. AB → BC → CA triangles within `Δt`) — predictive of information transfer.

Graph-neural networks then predict outcomes such as disease spread, fight initiation, or rank changes.

### 12.4a  Social-network analysis — extending to temporal motifs

Temporal motifs are repeated, short sequences of social interactions that can predict future behavior (e.g. rank change, fight outcome). Here we give a concrete implementation of a 3-node, 3-edge time-ordered triangle.

**Data:** A list of interactions `(t, src, tgt)` where `src` initiated an interaction with `tgt` at time `t` (e.g. approach, groom, attack), or undirected proximity events.

**Motif definition (3-node, 3-edge with time order):**

- At time t1: A → B
- At time t2 (t1 < t2 ≤ t1+Δt): B → C
- At time t3 (t2 < t3 ≤ t2+Δt): C → A

```python
import numpy as np
from collections import defaultdict

def find_temporal_triangles(interactions, delta_t=5.0):
    """
    interactions: list of (t, src, tgt) with t as float seconds.
    Returns list of (t_start, (A,B,C)) where the triangle began at t_start.
    """
    triangles = []
    # Create index by source for fast lookup
    by_src = defaultdict(list)
    for t, src, tgt in interactions:
        by_src[src].append((t, tgt))
    # For each A->B event, look for B->C and C->A within delta_t
    for t_ab, a, b in interactions:
        # B->C events after t_ab
        for t_bc, c in by_src.get(b, []):
            if t_bc < t_ab or t_bc > t_ab + delta_t:
                continue
            # C->A events after t_bc
            for t_ca, a2 in by_src.get(c, []):
                if a2 != a:
                    continue
                if t_ca < t_bc or t_ca > t_bc + delta_t:
                    continue
                triangles.append((t_ab, (a, b, c)))
    return triangles

# Count motifs per individual
motif_counts = defaultdict(int)
for t, (a, b, c) in find_temporal_triangles(interactions, delta_t=2.0):
    motif_counts[a] += 1  # count how often this mouse starts the triangle
```

Use the motif counts as features to predict dominance rank via logistic regression.

**Pitfall:** Temporal motifs are computationally expensive (O(N³) in the number of interactions). For large datasets (>1M interactions), use time-binned sampling or graph-based motif counting (e.g. `networkx.algorithms.temporal`).

## 12.5  Collective behavior

Schools, flocks, and swarms are exquisite testbeds for *interpretable* deep learning. A common protocol:

1. Track all individuals' velocity vectors.
2. Fit a Vicsek-like model with neural force terms.
3. Compare with the symbolic mechanistic baseline; the *residual* is the learnable part.

### 12.5a  Collective behavior — neural force terms for Vicsek-like models

In the Vicsek model, each individual's velocity aligns with its neighbors within radius R, plus noise. The neural extension replaces the alignment rule with a network that predicts the change in velocity from the local neighborhood state:

\\( v_i(t+\\Delta t) \\approx v_i(t) + \\Delta t \\, f_\\theta(\\text{neighborhood}_i(t)) \\)

**Data requirements:** Tracked trajectories of N individuals over time (positions, velocities).

**Neighborhood encoding:** For each individual i at time t, build a feature vector that is invariant to translation and rotation — the relative positions and velocities of neighbors expressed in a local frame aligned with `v_i`.

```python
import torch
import torch.nn as nn

class CollectiveForceNet(nn.Module):
    def __init__(self, input_dim=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)  # acceleration in x,y
        )

    def forward(self, x):
        return self.net(x)

def build_neighborhood_features(pos, vel, i, t, N, R=10.0):
    """
    pos: (N, T, 2)
    vel: (N, T, 2)
    Returns feature vector for individual i at time t.
    """
    x_i, y_i = pos[i, t]
    vx_i, vy_i = vel[i, t]
    # Find neighbors within distance R
    dists = np.sqrt((pos[:, t, 0] - x_i)**2 + (pos[:, t, 1] - y_i)**2)
    neighbors = np.where((dists < R) & (np.arange(N) != i))[0]
    if len(neighbors) == 0:
        # Use a zero vector
        return np.zeros(64)
    # Relative positions in body frame (rotate so that v_i points along x)
    angle = np.arctan2(vy_i, vx_i)
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    features = []
    for j in neighbors:
        dx = pos[j, t, 0] - x_i
        dy = pos[j, t, 1] - y_i
        rel_pos = rot @ np.array([dx, dy])
        rel_vel = rot @ (vel[j, t] - vel[i, t])
        features.extend([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1]])
    # Pad or truncate to fixed length (e.g. max 16 neighbors)
    if len(features) > 64:
        features = features[:64]
    else:
        features += [0] * (64 - len(features))
    return np.array(features)
```

**Training:** Collect `(input_feature, target_acceleration)` pairs from the data and train the network to predict acceleration. The residual from the Vicsek rule (if any) is the "learnable part."

**Pitfall:** Collective-motion data are highly autocorrelated. Use time-block cross-validation (train on the first 80% of time, test on the last 20%) to avoid overfitting.

## 12.6  Ethical considerations

- **Welfare.** Marker-less methods reduce surgical intervention; document IACUC / ethics approval.
- **Surveillance creep.** The same tools can be applied to humans; treat publication of identifiable animal IDs with care.
- **Wildlife disturbance.** Drone-mounted observation has measurable physiological cost for many species; use camera traps when possible.

### 12.6a  Extended pitfalls — ethical considerations beyond IACUC

The points above flag welfare, surveillance, and wildlife disturbance. Below are practical compliance steps for each.

**Welfare:**

- For markerless methods, still require IACUC approval if animals are housed or handled. Document that video recording does not cause distress (e.g. no bright lights, no prolonged restraint).
- For open-field arenas, ensure the environment is enriched (shelter, bedding) unless the scientific question requires deprivation.

**Surveillance creep:**

- If your method can track humans, do not release code or models that would lower the barrier for mass surveillance without robust access control.
- For de-identification of animal videos: remove ear tags or other identifiers from the video before publishing. Store raw videos on secured servers.

**Wildlife disturbance:**

- Drone flights over nesting colonies can cause stress and abandonment. Use recommended minimum altitudes (e.g. >50 m for seabirds). Validate with behavioral observations during pilot flights.
- If using camera traps, document the thermal and noise signatures; some species (e.g. felids) are trap-shy.

**Data sovereignty:** For animal behavior data collected on Indigenous lands or involving culturally significant species (e.g. eagles, wolves), consult with local communities before publication. Provide options for embargo or co-authorship.

**Pitfall:** Ethical review boards (IACUC, IRB) may not be familiar with AI-specific risks. Include a "Data Ethics" section in your protocol that explicitly addresses model misuse, re-identification, and data sharing.

## 12.7  Exercises

1. **DLC vs. SLEAP.** Annotate 200 frames of mouse video. Train DeepLabCut and SLEAP; compare keypoint RMSE on held-out frames.
2. **Syllable replication.** Apply Keypoint-MoSeq to a published open-field dataset. Reproduce the syllable count and average duration reported by the original authors within ± 20 %.
3. **Social GNN.** On the SOCIAL-Mice dataset, train a GNN to predict the dominance rank of an individual from one week of interactions.
4. **Welfare audit.** For the animal videos you used above, write a short welfare statement covering housing, IACUC protocol, and de-identification.

**Additional exercises:**

5. **(12.7f) Benchmark SLEAP vs. DLC for multi-mouse.** Use the publicly available CalMS21 dataset (multi-mouse social interactions). Train both SLEAP and DeepLabCut (with multi-animal extension) on 200 labeled frames. Compare identity tracking accuracy (multi-animal identity swap rate) and keypoint localization error (RMSE). Which performs better when mice overlap?
6. **(12.7g) Syllable stability across days.** Record the same mouse in an open field for 5 consecutive days. Run Keypoint-MoSeq on each day separately. Align syllables across days using hierarchical clustering of their mean trajectories. Compute the fraction of syllables that are "stable" (i.e. appear in ≥4 days). How does stability vary with syllable type (e.g. grooming vs. running)?
7. **(12.7h) Temporal motifs predict aggression.** In a mouse colony, record all social interactions (approach, sniff, attack). Detect temporal motifs (triangles) as above. Use motif counts per individual to predict which mouse will win the next agonistic encounter. Compare to a baseline using simple interaction counts.
8. **(12.7i) Neural Vicsek on fish-school data.** Download a publicly available fish tracking dataset (e.g. from the Collective Behavior repository). Train a neural force model using the features described above. Compare its long-term trajectory prediction error (10 steps ahead) to the classic Vicsek model (with optimized parameters). Does the neural model generalize to higher densities?

## 12.8  Further reading

- Mathis, A. *DeepLabCut.* Nat Neurosci (2018).
- Pereira, T. *SLEAP.* Nat. Methods (2022).
- Wiltschko, A. *Mapping sub-second structure in mouse behavior.* Neuron (2015) — MoSeq.
- Tuia, D. *Perspectives in machine learning for wildlife conservation.* Nat Commun (2022).
- Pereira, T. D. et al. *SLEAP: A deep learning system for multi-animal pose tracking.* Nat. Methods (2022) — includes a benchmark against DLC.
- Wiltschko, A. B. et al. *Revealing the structure of pharmacobehavioral space through motion sequencing.* Nat Neurosci (2020) — MoSeq applied to drug effects.
- Berman, G. J. *Measuring behavior across scales.* Curr Opin Neurobiol (2018) — review of behavior quantification.
- Marshall, J. D. et al. *Continuous whole-body 3D kinematic recordings across the rodent behavioral repertoire.* Neuron (2021) — dataset and methods.

## See also

- [Chapter 11 — Neuroscience](chapter_11_neuroscience.md)
- [Chapter 14 — Ecology & Conservation](chapter_14_ecology.md)
- [Chapter 19 — Ethics of AI in Biology](chapter_19_ethics.md)
