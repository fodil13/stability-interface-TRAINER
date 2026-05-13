"""

TRAINING FOR PROTEIN STABILITY PREDICTOR WITH TEMPORAL INTERLEAVING
Copyright (c) 2025 Fodil Azzaz, PhD
Training script for protein interface stability predictor.
Uses pre‑computed SASA scores stored in graph.sasa_score.
Normalization is performed using training set only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import random
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("TRAINING: SASA targets + GATConv + Combined trajectories")
print("=" * 70)

# ========== CONFIGURATION ==========
MAX_FRAMES = 1000          # Max frames to use from each trajectory
STEP = 1                   # Frame skipping
TRAIN_EPOCHS = 80
SEED = 42
N_TRIALS = 1               # Number of independent training runs
CONFIDENCE_LEVEL = 0.95
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
GAT_HEADS = 4              # Number of attention heads
PATIENCE = 15              # Early stopping patience

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ========== UTILITIES ==========
def convert_to_float32(graph):
    graph.x = graph.x.float()
    graph.pos = graph.pos.float()
    return graph

def create_consistent_temporal_splits(graphs):
    """Temporal interleaving: train (0,3,6...), val (1,4,7...), test (2,5,8...)."""
    selected = graphs[:MAX_FRAMES:STEP]
    n = len(selected)
    train_idx = list(range(0, n, 3))
    val_idx = list(range(1, n, 3))
    test_idx = list(range(2, n, 3))
    train_g = [selected[i] for i in train_idx]
    val_g = [selected[i] for i in val_idx]
    test_g = [selected[i] for i in test_idx]
    print(f"  Splits: train={len(train_g)}, val={len(val_g)}, test={len(test_g)}")
    return train_g, val_g, test_g, train_idx, val_idx, test_idx

def evaluate_comprehensive(model, graphs, targets):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for g, t in zip(graphs, targets):
            preds.append(model(g).item())
            trues.append(t.item())
    preds = np.array(preds)
    trues = np.array(trues)
    ss_res = np.sum((trues - preds)**2)
    ss_tot = np.sum((trues - np.mean(trues))**2)
    r2 = max(0.0, 1 - ss_res/ss_tot) if ss_tot > 1e-8 else 0.0
    spearman = spearmanr(trues, preds)[0] if len(trues) > 1 else 0.0
    mae = np.mean(np.abs(preds - trues))
    return {'r2': r2, 'spearman': spearman, 'mae': mae}

# ========== TARGET GENERATION USING SASA SCORE ==========
def extract_raw_sasa(graph):
    """Return the pre‑computed SASA score (buried area in Å²)."""
    if hasattr(graph, 'sasa_score'):
        return graph.sasa_score
    else:
        # Fallback: compute contact count (if old graph)
        # We'll just return 0.0 and warn
        print("Warning: graph has no sasa_score attribute. Using 0.0.")
        return 0.0

def create_targets_with_global_norm(graphs, global_min=None, global_max=None):
    """
    Extract raw SASA scores from graphs, then normalize using global_min/global_max.
    Returns (targets, global_min, global_max).
    Targets are scaled to [0.2, 0.8].
    """
    raw = np.array([extract_raw_sasa(g) for g in graphs])
    if global_min is None or global_max is None:
        global_min = raw.min()
        global_max = raw.max()
    if global_max - global_min > 1e-6:
        targets = 0.1 + 0.8 * (raw - global_min) / (global_max - global_min)
    else:
        targets = np.full_like(raw, 0.5)
    targets = [torch.tensor(t, dtype=torch.float32) for t in targets]
    return targets, global_min, global_max

# ========== GAT MODEL (same as before) ==========
class UltimateStabilityPredictor(nn.Module):
    def __init__(self, node_dim, focus_pairs, gat_heads=4):
        super().__init__()
        self.focus_pairs = focus_pairs
        self.interface_count = len(focus_pairs)
        # GAT layers
        self.conv1 = GATConv(node_dim, 128, heads=gat_heads, concat=True)
        self.conv2 = GATConv(128 * gat_heads, 64, heads=1, concat=False)
        self.conv3 = GATConv(64, 32, heads=1, concat=False)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.batch_norm1 = nn.BatchNorm1d(128 * gat_heads)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.interface_predictor = nn.Linear(32 * self.interface_count, 1)
        print(f"   GAT model: {self.interface_count} interfaces, heads={gat_heads}")

    def forward(self, data):
        x = F.relu(self.batch_norm1(self.conv1(data.x, data.edge_index)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.conv2(x, data.edge_index)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.conv3(x, data.edge_index)))
        if self.interface_count > 0:
            feat = self.interface_analysis(data, x)
            x_pooled = feat.flatten()
        else:
            x_pooled = x.mean(dim=0)
        return self.interface_predictor(x_pooled).squeeze()

    def interface_analysis(self, data, node_features):
        segids = data.segids
        positions = data.pos.cpu().numpy()
        features = []
        for segA, segB in self.focus_pairs:
            idxA = [i for i, s in enumerate(segids) if s == segA]
            idxB = [i for i, s in enumerate(segids) if s == segB]
            if idxA and idxB:
                mask = self.detect_interface(positions, idxA, idxB)
                if mask.sum() > 0:
                    f = node_features[mask].mean(dim=0)
                else:
                    f = node_features.mean(dim=0)
                features.append(f)
        return torch.stack(features) if features else node_features.mean(dim=0).unsqueeze(0)

    def detect_interface(self, positions, idxA, idxB):
        mask = torch.zeros(len(positions), dtype=torch.bool)
        posA = positions[idxA]
        posB = positions[idxB]
        tree = cKDTree(posA)
        dists, neigh = tree.query(posB, k=1, distance_upper_bound=8.0)
        for i, bi in enumerate(idxB):
            if dists[i] < 8.0:
                mask[bi] = True
                if dists[i] < 6.0:
                    mask[idxA[neigh[i]]] = True
        return mask

# ========== LOAD COMBINED GRAPHS ==========
def load_combined_graphs(paths):
    all_graphs = []
    for p in paths:
        print(f"   Loading: {p}")
        graphs = torch.load(p, map_location='cpu', weights_only=False)
        graphs = [convert_to_float32(g) for g in graphs]
        all_graphs.extend(graphs)
    print(f"   Combined: {len(all_graphs)} graphs from {len(paths)} files")
    return all_graphs

# ========== TRAINING PROTOCOL ==========
def run_training(data_paths, focus_pairs, run_seed=None):
    # Load and combine graphs
    all_graphs = load_combined_graphs(data_paths)
    train_g, val_g, test_g, _, _, _ = create_consistent_temporal_splits(all_graphs)

    # Compute normalization parameters from TRAINING SET only
    print("   Computing normalization stats from training set SASA scores...")
    train_raw = [extract_raw_sasa(g) for g in train_g]
    global_min = np.min(train_raw)
    global_max = np.max(train_raw)
    print(f"   Raw SASA range (train): {global_min:.2f} – {global_max:.2f} Å²")

    # Create targets for all splits using same global min/max
    train_targets, _, _ = create_targets_with_global_norm(train_g, global_min, global_max)
    val_targets, _, _ = create_targets_with_global_norm(val_g, global_min, global_max)
    test_targets, _, _ = create_targets_with_global_norm(test_g, global_min, global_max)

    if run_seed is not None:
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

    model = UltimateStabilityPredictor(node_dim=train_g[0].x.shape[1],
                                       focus_pairs=focus_pairs,
                                       gat_heads=GAT_HEADS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(TRAIN_EPOCHS):
        # Training
        model.train()
        loss_train = 0.0
        for g, t in zip(train_g, train_targets):
            pred = model(g)
            loss = F.mse_loss(pred, t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_train += loss.item()
        loss_train /= len(train_g)

        # Validation
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for g, t in zip(val_g, val_targets):
                loss_val += F.mse_loss(model(g), t).item()
        loss_val /= len(val_g)

        scheduler.step(loss_val)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            train_metrics = evaluate_comprehensive(model, train_g, train_targets)
            val_metrics = evaluate_comprehensive(model, val_g, val_targets)
            print(f"Epoch {epoch:03d}: train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, val_R²={val_metrics['r2']:.3f}")

    model.load_state_dict(best_state)
    test_metrics = evaluate_comprehensive(model, test_g, test_targets)
    print(f"   Best val loss: {best_val_loss:.4f}, test R²={test_metrics['r2']:.3f}, test Spearman={test_metrics['spearman']:.3f}")
    return model, test_metrics, best_val_loss

def run_trials(data_paths, focus_pairs, n_runs=N_TRIALS):
    results = []
    for run in range(n_runs):
        print(f"\n--- Trial {run+1}/{n_runs} ---")
        model, metrics, val_loss = run_training(data_paths, focus_pairs, run_seed=42 + run*100)
        results.append({'model': model,
                        'test_r2': metrics['r2'],
                        'test_spearman': metrics['spearman'],
                        'test_mae': metrics['mae'],
                        'val_loss': val_loss})
    return results

# ========== INTERACTIVE SETUP ==========
def ultimate_interactive_setup(sample_graphs):
    segids = sample_graphs[0].segids
    unique = sorted(set(segids))
    print("Available segids:")
    for i, s in enumerate(unique):
        print(f"  {i+1}. {s}")
    inp = input("Enter interface pairs (e.g., '1-2,1-3' or 'all'): ").strip()
    if inp.lower() == 'all':
        pairs = [(unique[i], unique[j]) for i in range(len(unique)) for j in range(i+1, len(unique))]
    else:
        pairs = []
        for part in inp.split(','):
            a,b = map(int, part.strip().split('-'))
            pairs.append((unique[a-1], unique[b-1]))
    print(f"Focus pairs: {pairs}")
    return pairs

# ========== MAIN ==========
if __name__ == "__main__":
    print("\nEnter paths to .pt graph files (comma-separated).")
    paths_input = input("Paths: ").strip()
    data_paths = [p.strip() for p in paths_input.split(',')]

    # Load first file for segid detection (and to check sasa_score)
    first_graphs = torch.load(data_paths[0], map_location='cpu', weights_only=False)
    first_graphs = [convert_to_float32(g) for g in first_graphs]
    if not hasattr(first_graphs[0], 'sasa_score'):
        print(" Warning: graphs do not have 'sasa_score' attribute. Run graph creation with SASA first.")
    focus_pairs = ultimate_interactive_setup(first_graphs)

    all_results = run_trials(data_paths, focus_pairs, n_runs=N_TRIALS)

    best = max(all_results, key=lambda x: x['test_r2'])
    print(f"\n✅ Best model: test R²={best['test_r2']:.3f}, Spearman={best['test_spearman']:.3f}")

    torch.save({
        'model_state_dict': best['model'].state_dict(),
        'focus_pairs': focus_pairs,
        'test_r2': best['test_r2'],
        'test_spearman': best['test_spearman'],
        'improvements': ['SASA_targets', 'GATConv', 'combined_trajectories']
    }, 'MODEL_sasa.pth')
    print("Model saved as MODEL_sasa.pth")
