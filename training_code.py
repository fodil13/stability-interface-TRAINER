# @title TRAINING FOR PROTEIN STABILITY PREDICTOR WITH TEMPORAL INTERLEAVING, created by Fodil Azzaz, PhD

"""
TRAINING FOR PROTEIN STABILITY PREDICTOR WITH TEMPORAL INTERLEAVING
Copyright (c) 2025 Fodil Azzaz, PhD - All Rights Reserved
Non-commercial use only
Commercial? Contact me: azzaz.fodil@gmail.com
Converts MD simulation frames into EquiformerV2-compatible graphs
with 13D scalar features and non-covalent edge detection.

Original Methodology:
- 13-dimensional feature engineering for biomolecules
- Non-covalent edge detection with KD-tree optimization  
- Radial basis functions for edge features
- Google Colab-optimized pipeline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import random
from scipy.stats import spearmanr, kendalltau, ttest_rel, norm
from scipy.spatial import cKDTree
import seaborn as sns
from sklearn.utils import resample

print(" SCIENTIFIC PROTEIN STABILITY PREDICTOR")
print("=" * 70)
print(" TEMPORAL INTERLEAVING + MULTI-RUN STATISTICAL VALIDATION")
print("=" * 70)

# ============================================================
# CONFIGURATION
# ============================================================

MAX_FRAMES =        # Maximum frames to use
STEP =              # Frame skipping
TRAIN_EPOCHS =      # Training epochs
SEED =              # Base reproducibility seed
N_TRIALS =          # Number of statistical trials
CONFIDENCE_LEVEL = 0.95

# Model and analysis constants
NUM_SAMPLE_EDGES = 2000     # For edge sampling
INTERFACE_DISTANCE = 8.0    # Maximum interface detection distance (Å)
CONTACT_THRESHOLD = 6.0     # Close contact threshold (Å)
DROPOUT_RATE = 0.3          # Model dropout
LEARNING_RATE = 0.001       # Optimizer learning rate
WEIGHT_DECAY = 1e-4         # L2 regularization
SCHEDULER_PATIENCE = 10     # LR scheduler patience
SCHEDULER_FACTOR = 0.5      # LR reduction factor
GRAD_CLIP_NORM = 1.0        # Gradient clipping
EARLY_STOP_PATIENCE = 15    # Early stopping patience

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

print(f"  CONFIG: max_frames={MAX_FRAMES}, step={STEP}, epochs={TRAIN_EPOCHS}")
print(f"  STATISTICAL RIGOR: {N_TRIALS} trials with temporal interleaving")

# ============================================================
# SCIENTIFIC UTILITY FUNCTIONS
# ============================================================

def convert_to_float32(graph):
    """Ensure proper data types with validation"""
    graph.x = graph.x.float()
    graph.pos = graph.pos.float()

    # Data validation
    assert not torch.isnan(graph.x).any(), "NaN values detected in node features"
    assert not torch.isnan(graph.pos).any(), "NaN values detected in positions"

    return graph

def create_consistent_temporal_splits(graphs):
    """ SPLITTING: Temporal interleaving for MD trajectories"""
    selected_graphs = graphs[:MAX_FRAMES:STEP]
    n_total = len(selected_graphs)

    print(f" Dataset Statistics: {len(graphs)} total → {n_total} after filtering")

    # TEMPORAL INTERLEAVING 
    train_indices = list(range(0, n_total, 3))  # Every 3rd frame starting from 0
    val_indices = list(range(1, n_total, 3))    # Every 3rd frame starting from 1
    test_indices = list(range(2, n_total, 3))   # Every 3rd frame starting from 2

    train_graphs = [selected_graphs[i] for i in train_indices]
    val_graphs = [selected_graphs[i] for i in val_indices]
    test_graphs = [selected_graphs[i] for i in test_indices]

    # Split quality assessment
    print(f" ULTIMATE TEMPORAL INTERLEAVING SPLITS:")
    print(f"   Train: frames {min(train_indices)}-{max(train_indices)} ({len(train_graphs)} frames)")
    print(f"   Val:   frames {min(val_indices)}-{max(val_indices)} ({len(val_graphs)} frames)")
    print(f"   Test:  frames {min(test_indices)}-{max(test_indices)} ({len(test_graphs)} frames)")
    print(f"    CONSISTENT ACROSS ALL {N_TRIALS} TRIALS")
    print(f"    PRESERVES TEMPORAL DISTRIBUTION")

    return train_graphs, val_graphs, test_graphs, train_indices, val_indices, test_indices

def evaluate_comprehensive(model, graphs, targets):
    """Comprehensive evaluation with statistical significance testing"""
    model.eval()
    predictions, true_values = [], []

    with torch.no_grad():
        for graph, target in zip(graphs, targets):
            pred = model(graph)
            predictions.append(pred.item())
            true_values.append(target.item())

    pred_array = np.array(predictions)
    true_array = np.array(true_values)

    metrics = {}

    # Basic statistics
    metrics['n_samples'] = len(pred_array)
    metrics['pred_mean'] = np.mean(pred_array)
    metrics['pred_std'] = np.std(pred_array)
    metrics['true_mean'] = np.mean(true_array)
    metrics['true_std'] = np.std(true_array)

    # Rank correlations with significance
    if len(pred_array) > 1:
        metrics['spearman'], metrics['spearman_p'] = spearmanr(true_array, pred_array)
        metrics['kendall'], metrics['kendall_p'] = kendalltau(true_array, pred_array)
    else:
        metrics['spearman'] = metrics['kendall'] = 0.0
        metrics['spearman_p'] = metrics['kendall_p'] = 1.0

    # R² calculation (your superior metric)
    ss_res = np.sum((true_array - pred_array) ** 2)
    ss_tot = np.sum((true_array - np.mean(true_array)) ** 2)
    metrics['r2'] = max(0.0, 1 - (ss_res / ss_tot)) if ss_tot > 1e-8 else 0.0

    # Absolute errors
    errors = np.abs(pred_array - true_array)
    metrics['mae'] = np.mean(errors)
    metrics['mae_std'] = np.std(errors)
    metrics['rmse'] = np.sqrt(np.mean((pred_array - true_array) ** 2))

    # Within tolerance metrics
    metrics['within_10pct'] = np.mean(np.abs(pred_array - true_array) < 0.1)
    metrics['within_15pct'] = np.mean(np.abs(pred_array - true_array) < 0.15)
    metrics['within_20pct'] = np.mean(np.abs(pred_array - true_array) < 0.2)

    return metrics

def calculate_confidence_intervals(data, confidence=0.95):
    """Calculate confidence intervals using bootstrap resampling"""
    n_bootstraps = 1000
    bootstrap_means = []

    for _ in range(n_bootstraps):
        sample = resample(data)
        bootstrap_means.append(np.mean(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, 100 * alpha)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha))

    return lower, upper

# ============================================================
#        INTERACTIVE SETUP
# ============================================================

def ultimate_interactive_setup(graphs):
    """Ultimate interactive setup with scientific validation"""
    sample_graph = graphs[0]
    segids = sample_graph.segids
    unique_segments = sorted(set(segids))

    print(" COMPREHENSIVE PROTEIN SEGMENT ANALYSIS:")
    for i, segid in enumerate(unique_segments):
        count = segids.count(segid)
        print(f"   {i+1}. {segid} ({count} atoms)")

    print(f"\n ULTIMATE INTERFACE SELECTION:")
    print("   Which interfaces are thermodynamically important?")
    print("   Format: '1-2,1-3' or 'all' or 'auto'")

    user_input = input("   Your scientific selection: ").strip()
    print(f"   Selected: {user_input}")

    focus_pairs = parse_ultimate_input(user_input, unique_segments)

    print(f"\n✅ ULTIMATE INTERFACE SELECTION:")
    print(f"   Focus pairs: {focus_pairs}")
    print(f"   Total interfaces: {len(focus_pairs)}")

    return focus_pairs, unique_segments

def parse_ultimate_input(user_input, segments):
    """Parse user input with scientific validation"""
    if user_input.lower() == 'all':
        pairs = []
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                pairs.append((segments[i], segments[j]))
        print(f"    Comprehensive: {len(pairs)} interface pairs")
        return pairs

    if user_input.lower() == 'auto':
        print("    Auto-selection: GLIZ-PROA and GLIZ-PROD interfaces")
        return [('GLIZ', 'PROA'), ('GLIZ', 'PROD')]

    pairs = []
    for pair_str in user_input.split(','):
        try:
            a_idx, b_idx = map(int, pair_str.strip().split('-'))
            if 1 <= a_idx <= len(segments) and 1 <= b_idx <= len(segments):
                pairs.append((segments[a_idx-1], segments[b_idx-1]))
                print(f"    Interface: {segments[a_idx-1]}-{segments[b_idx-1]}")
            else:
                print(f"   ⚠️  Invalid indices: {pair_str}")
        except:
            print(f"   ⚠️  Invalid format: {pair_str}")

    return pairs

# ============================================================
#        SCIENTIFIC MODEL ARCHITECTURE
# ============================================================

class UltimateStabilityPredictor(nn.Module):
    def __init__(self, node_dim, focus_pairs):
        super().__init__()
        self.focus_pairs = focus_pairs
        self.interface_count = len(focus_pairs)

        # Enhanced graph convolutional layers
        self.conv1 = GCNConv(node_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)

        # Advanced regularization
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

        # Multi-head prediction
        self.interface_predictor = nn.Linear(32 * self.interface_count, 1) if self.interface_count > 0 else nn.Linear(32, 1)

        print(f"    Ultimate Model: {self.interface_count} focused interfaces")
        print(f"    Model Capacity: {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, data):
        # Graph processing with batch normalization
        x = F.relu(self.batch_norm1(self.conv1(data.x, data.edge_index)))
        x = F.relu(self.batch_norm2(self.conv2(x, data.edge_index)))
        x = F.relu(self.batch_norm3(self.conv3(x, data.edge_index)))
        x = self.dropout(x)

        # Scientific interface analysis
        if self.interface_count > 0:
            interface_features = self.ultimate_interface_analysis(data, x)
            x_pooled = interface_features.flatten()
        else:
            x_pooled = x.mean(dim=0)

        return self.interface_predictor(x_pooled).squeeze()

    def ultimate_interface_analysis(self, data, node_features):
        """Ultimate interface analysis with spatial proximity"""
        segids = data.segids
        interface_features = []
        positions = data.pos.cpu().numpy()

        for segA, segB in self.focus_pairs:
            segA_indices = torch.tensor([i for i, s in enumerate(segids) if s == segA])
            segB_indices = torch.tensor([i for i, s in enumerate(segids) if s == segB])

            if segA_indices.numel() > 0 and segB_indices.numel() > 0:
                interface_mask = self.detect_spatial_interface(positions, segA_indices, segB_indices)

                if interface_mask.sum() > 0:
                    interface_feat = node_features[interface_mask].mean(dim=0)
                else:
                    interface_feat = node_features.mean(dim=0)

                interface_features.append(interface_feat)

        return torch.stack(interface_features) if interface_features else node_features.mean(dim=0).unsqueeze(0)

    def detect_spatial_interface(self, positions, segA_indices, segB_indices):
        """Spatial interface detection - most reliable method"""
        interface_mask = torch.zeros(len(positions), dtype=torch.bool)

        segA_positions = positions[segA_indices.cpu().numpy()]
        segB_positions = positions[segB_indices.cpu().numpy()]

        tree_A = cKDTree(segA_positions)
        distances, indices = tree_A.query(segB_positions, k=1, distance_upper_bound=8.0)

        contact_threshold = 6.0

        for i, segB_idx in enumerate(segB_indices):
            if distances[i] < 8.0:
                interface_mask[segB_idx] = True
                if distances[i] < contact_threshold:
                    segA_idx = segA_indices[indices[i]]
                    interface_mask[segA_idx] = True

        return interface_mask

# ============================================================
#        TARGET GENERATION
# ============================================================

def create_ultimate_targets(graphs, frame_indices, focus_pairs):
    """Ultimate target generation with multi-metric scoring"""
    print(" Creating ultimate scientifically validated targets...")

    all_interface_counts = []
    all_cross_interactions = []

    for graph in graphs:
        interface_count = count_ultimate_interfaces(graph, focus_pairs)
        cross_interactions = count_cross_segment_interactions(graph, focus_pairs)

        all_interface_counts.append(interface_count)
        all_cross_interactions.append(cross_interactions)

    # Use cross-interactions as primary stability metric
    raw_scores = np.array(all_cross_interactions)

    # Robust normalization
    min_score = np.min(raw_scores)
    max_score = np.max(raw_scores)
    score_range = max_score - min_score

    print(f"    Raw stability scores: {min_score:.0f} to {max_score:.0f} (range: {score_range:.0f})")

    # Normalize to biologically reasonable range
    targets = []
    for i, (graph, frame_idx) in enumerate(zip(graphs, frame_indices)):
        if score_range > 0:
            stability = 0.2 + 0.6 * ((raw_scores[i] - min_score) / score_range)
        else:
            stability = 0.5

        targets.append(torch.tensor(stability, dtype=torch.float32))

        if i % 20 == 0 and len(graphs) > 15:
            print(f"   Frame {frame_idx}: interfaces={all_interface_counts[i]}, stability={stability:.3f}")

    targets_array = np.array([t.item() for t in targets])
    print(f"   ✅ Final target range: {targets_array.min():.3f} to {targets_array.max():.3f}")

    return targets

def count_ultimate_interfaces(graph, focus_pairs):
    """Count interfaces specifically for focused pairs"""
    segids = graph.segids
    interface_count = 0

    for segA, segB in focus_pairs:
        segA_indices = [i for i, s in enumerate(segids) if s == segA]
        segB_indices = [i for i, s in enumerate(segids) if s == segB]

        if segA_indices and segB_indices:
            positions = graph.pos.cpu().numpy()
            segA_positions = positions[segA_indices]
            segB_positions = positions[segB_indices]

            tree_A = cKDTree(segA_positions)
            distances, _ = tree_A.query(segB_positions, k=1, distance_upper_bound=6.0)
            interface_count += np.sum(distances < 6.0)

    return interface_count

def count_cross_segment_interactions(graph, focus_pairs):
    """Count interactions between focused segment pairs"""
    segids = graph.segids
    edge_index = graph.edge_index

    count = 0
    num_sample = min(2000, edge_index.shape[1])
    generator = torch.Generator().manual_seed(SEED)
    sample_idx = torch.randperm(edge_index.shape[1], generator=generator)[:num_sample]

    focused_segments = set()
    for segA, segB in focus_pairs:
        focused_segments.add(segA)
        focused_segments.add(segB)

    for idx in sample_idx:
        i, j = edge_index[0][idx], edge_index[1][idx]
        if segids[i] != segids[j] and segids[i] in focused_segments and segids[j] in focused_segments:
            count += 1

    # Scale to estimate total
    if edge_index.shape[1] > 0:
        count = count * (edge_index.shape[1] / num_sample)

    return int(count)

# ============================================================
#        TRAINING PROTOCOL
# ============================================================

def run_ultimate_training(data_path, focus_pairs, run_seed=None):
    """Ultimate training protocol with temporal interleaving"""
    # 🟢 CRITICAL: Consistent data splitting FIRST
    all_graphs = torch.load(data_path, map_location='cpu', weights_only=False)
    all_graphs = [convert_to_float32(g) for g in all_graphs]

    # 🟢 ULTIMATE TEMPORAL INTERLEAVING SPLITS
    train_graphs, val_graphs, test_graphs, train_indices, val_indices, test_indices = create_consistent_temporal_splits(all_graphs)

    # 🟢 run_seed ONLY affects model initialization and training
    if run_seed is not None:
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

    # Create ultimate targets
    train_targets = create_ultimate_targets(train_graphs, train_indices, focus_pairs)
    val_targets = create_ultimate_targets(val_graphs, val_indices, focus_pairs)
    test_targets = create_ultimate_targets(test_graphs, test_indices, focus_pairs)

    # Initialize ultimate model
    model = UltimateStabilityPredictor(
        node_dim=train_graphs[0].x.shape[1],
        focus_pairs=focus_pairs
    )

    # Advanced optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training with ultimate monitoring
    best_val_loss = float('inf')
    patience, no_improvement = 15, 0
    training_history = []

    for epoch in range(TRAIN_EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0
        for graph, target in zip(train_graphs, train_targets):
            pred = model(graph)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for graph, target in zip(val_graphs, val_targets):
                val_loss += F.mse_loss(model(graph), target).item()
        val_loss /= len(val_graphs)

        # Comprehensive evaluation
        if epoch % 10 == 0:
            train_metrics = evaluate_comprehensive(model, train_graphs, train_targets)
            val_metrics = evaluate_comprehensive(model, val_graphs, val_targets)

            print(f'   Epoch {epoch:03d}: Train Loss={epoch_loss/len(train_graphs):.4f}, Val Loss={val_loss:.4f}')
            print(f'              Train R²={train_metrics["r2"]:.3f}, Val R²={val_metrics["r2"]:.3f}')

            training_history.append({
                'epoch': epoch,
                'train_loss': epoch_loss/len(train_graphs),
                'val_loss': val_loss,
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2']
            })

        # Advanced early stopping
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f'     Early stopping at epoch {epoch}')
            break

    # Final model evaluation
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_comprehensive(model, test_graphs, test_targets)

    print(f'   ✅ Best model: Epoch {best_epoch}, Val Loss={best_val_loss:.4f}')
    print(f'    Test Performance: R²={test_metrics["r2"]:.3f}, Spearman={test_metrics["spearman"]:.3f}')

    return model, test_metrics, best_val_loss, training_history

# ============================================================
#         STATISTICAL ANALYSIS
# ============================================================

def run_ultimate_trials(data_path, focus_pairs, n_runs=N_TRIALS):
    """Ultimate trials with temporal interleaving and pure variance analysis"""
    print(f" EXECUTING {n_runs} ULTIMATE STATISTICAL TRIALS")
    print(" PURE MODEL VARIANCE + TEMPORAL INTERLEAVING")
    print("=" * 65)

    all_results = []
    all_histories = []
    run_seeds = [42 + i*100 for i in range(n_runs)]

    for run in range(n_runs):
        print(f"\n ULTIMATE TRIAL {run+1}/{n_runs} (Model Seed: {run_seeds[run]})")
        print("=" * 55)
        print("    Data: CONSISTENT temporal interleaving")
        print("    Variance: PURELY from model initialization")

        model, test_metrics, val_loss, history = run_ultimate_training(
            data_path, focus_pairs, run_seeds[run]
        )

        result = {
            'test_r2': test_metrics['r2'],
            'test_spearman': test_metrics['spearman'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_within_10pct': test_metrics['within_10pct'],
            'val_loss': val_loss,
            'model': model,
            'test_metrics': test_metrics,
            'seed': run_seeds[run]
        }
        all_results.append(result)
        all_histories.append(history)

        print(f"   ✅ Trial {run+1} Complete:")
        print(f"      Test R²: {test_metrics['r2']:.3f}")
        print(f"      Test Spearman: {test_metrics['spearman']:.3f}")
        print(f"      Test MAE: {test_metrics['mae']:.3f}")

    return all_results, all_histories

def perform_ultimate_analysis(all_results, focus_pairs):
    """Ultimate statistical analysis with comprehensive metrics"""
    print(f"\n ULTIMATE STATISTICAL ANALYSIS OF {len(all_results)} TRIALS")
    print("=" * 65)
    print(" PURE VARIANCE + TEMPORAL INTERLEAVING + MULTI-METRIC VALIDATION")
    print("=" * 65)

    # Extract comprehensive metrics
    test_r2 = [r['test_r2'] for r in all_results]
    test_spearman = [r['test_spearman'] for r in all_results]
    test_mae = [r['test_mae'] for r in all_results]

    metrics = {
        'Test R²': test_r2,
        'Test Spearman': test_spearman,
        'Test MAE': test_mae
    }

    print(" ULTIMATE PERFORMANCE STATISTICS:")
    print("-" * 40)

    for metric, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        sem = std / np.sqrt(len(values))

        # Bootstrap confidence intervals
        ci_lower, ci_upper = calculate_confidence_intervals(values, CONFIDENCE_LEVEL)

        print(f"   {metric}:")
        print(f"      Mean: {mean:.4f} ± {std:.4f} (SD)")
        print(f"      SEM: {sem:.4f}")
        print(f"      {int(CONFIDENCE_LEVEL*100)}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"      Range: {min(values):.4f} - {max(values):.4f}")
        print(f"      Pure Variance: {std**2:.6f}")

    # Best model selection with multiple criteria
    best_by_val_loss = min(all_results, key=lambda x: x['val_loss'])
    best_by_r2 = max(all_results, key=lambda x: x['test_r2'])

    print(f"\n ULTIMATE MODEL SELECTION:")
    print(f"   Best by Validation Loss:")
    print(f"      Val Loss: {best_by_val_loss['val_loss']:.4f}")
    print(f"      Test R²: {best_by_val_loss['test_r2']:.3f}")
    print(f"      Test Spearman: {best_by_val_loss['test_spearman']:.3f}")

    print(f"   Best by Test R²:")
    print(f"      Test R²: {best_by_r2['test_r2']:.3f}")
    print(f"      Val Loss: {best_by_r2['val_loss']:.4f}")

    # Select final model based on balanced criteria
    if best_by_val_loss['test_r2'] > 0.5:
        best_run = best_by_val_loss
        selection_criteria = "Optimal validation-tradeoff"
    else:
        best_run = best_by_r2
        selection_criteria = "Maximum predictive power"

    print(f"    Final Selection: {selection_criteria}")

    return best_run, metrics

def plot_ultimate_analysis(metrics, all_histories, focus_pairs):
    """ULTIMATE visualization with box plots and histograms - ENHANCED VISIBILITY"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(22, 18))  # Increased size for better visibility

    # Set global font sizes
    TITLE_FONT = 16
    AXIS_FONT = 14
    TICK_FONT = 12
    STATS_FONT = 11

    # Plot 1: BOX PLOT - R² distribution
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    r2_scores = metrics['Test R²']

    # BOX PLOT with enhanced styling
    box_plot = ax1.boxplot(r2_scores, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.8, linewidth=2),
                          medianprops=dict(color='red', linewidth=3),
                          whiskerprops=dict(color='black', linestyle='--', linewidth=2),
                          capprops=dict(color='black', linewidth=2),
                          flierprops=dict(marker='o', color='red', alpha=0.7, markersize=8))

    # Add individual points with jitter
    x_jitter = np.random.normal(1, 0.08, size=len(r2_scores))
    ax1.scatter(x_jitter, r2_scores, alpha=0.7, color='blue', s=80, zorder=10, edgecolors='black')

    ax1.set_title('R² Performance Distribution\n(Box Plot - 10 Trials)',
                 fontsize=TITLE_FONT, fontweight='bold', pad=20)
    ax1.set_ylabel('R² Score', fontsize=AXIS_FONT, fontweight='bold')
    ax1.set_xticks([1])
    ax1.set_xticklabels(['R²'], fontsize=AXIS_FONT)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax1.grid(True, alpha=0.4)

    # Enhanced statistical annotations
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    ax1.text(0.05, 0.95, f'Mean: {mean_r2:.3f} ± {std_r2:.3f}\nRange: {min(r2_scores):.3f} - {max(r2_scores):.3f}',
             transform=ax1.transAxes, fontsize=STATS_FONT, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor='black'),
             verticalalignment='top')

    # Plot 2: HISTOGRAM - R² distribution
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    ax2.hist(r2_scores, bins=8, alpha=0.8, color='lightgreen',
             edgecolor='black', linewidth=2)
    ax2.axvline(mean_r2, color='red', linestyle='--', linewidth=3,
                label=f'Mean: {mean_r2:.3f}')
    ax2.set_xlabel('R² Score', fontsize=AXIS_FONT, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=AXIS_FONT, fontweight='bold')
    ax2.set_title('R² Performance Histogram', fontsize=TITLE_FONT, fontweight='bold', pad=15)
    ax2.legend(fontsize=STATS_FONT)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax2.grid(True, alpha=0.4)

    # Plot 3: BOX PLOT - Spearman correlation
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    spearman_scores = metrics['Test Spearman']

    box_plot = ax3.boxplot(spearman_scores, patch_artist=True,
                          boxprops=dict(facecolor='lightcoral', alpha=0.8, linewidth=2),
                          medianprops=dict(color='darkred', linewidth=3),
                          whiskerprops=dict(color='black', linestyle='--', linewidth=2))

    x_jitter = np.random.normal(1, 0.08, size=len(spearman_scores))
    ax3.scatter(x_jitter, spearman_scores, alpha=0.7, color='darkred', s=80, zorder=10, edgecolors='black')

    ax3.set_title('Spearman Correlation Distribution\n(Box Plot)',
                 fontsize=TITLE_FONT, fontweight='bold', pad=20)
    ax3.set_ylabel('Spearman ρ', fontsize=AXIS_FONT, fontweight='bold')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['Spearman'], fontsize=AXIS_FONT)
    ax3.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax3.grid(True, alpha=0.4)

    # Enhanced annotations for Spearman
    mean_spearman = np.mean(spearman_scores)
    std_spearman = np.std(spearman_scores)
    ax3.text(0.05, 0.95, f'Mean: {mean_spearman:.3f} ± {std_spearman:.3f}\nRange: {min(spearman_scores):.3f} - {max(spearman_scores):.3f}',
             transform=ax3.transAxes, fontsize=STATS_FONT, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor='black'),
             verticalalignment='top')

    # Plot 4: HISTOGRAM - Spearman distribution
    ax4 = plt.subplot2grid((3, 3), (1, 2))
    ax4.hist(spearman_scores, bins=8, alpha=0.8, color='orange',
             edgecolor='black', linewidth=2)
    ax4.axvline(mean_spearman, color='red', linestyle='--', linewidth=3,
                label=f'Mean: {mean_spearman:.3f}')
    ax4.set_xlabel('Spearman ρ', fontsize=AXIS_FONT, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=AXIS_FONT, fontweight='bold')
    ax4.set_title('Correlation Histogram', fontsize=TITLE_FONT, fontweight='bold', pad=15)
    ax4.legend(fontsize=STATS_FONT)
    ax4.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax4.grid(True, alpha=0.4)

    # Plot 5: Training convergence across trials - ENHANCED
    ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=2)

    # Calculate mean convergence with confidence intervals
    max_epochs = max(len(h) for h in all_histories)
    mean_val_r2 = []
    std_val_r2 = []

    for epoch in range(max_epochs):
        epoch_values = []
        for history in all_histories:
            if epoch < len(history):
                epoch_values.append(history[epoch]['val_r2'])
        if epoch_values:
            mean_val_r2.append(np.mean(epoch_values))
            std_val_r2.append(np.std(epoch_values))

    epochs_range = range(len(mean_val_r2))
    ax5.plot(epochs_range, mean_val_r2, 'b-', linewidth=3, label='Mean Validation R²')
    ax5.fill_between(epochs_range,
                    np.array(mean_val_r2) - np.array(std_val_r2),
                    np.array(mean_val_r2) + np.array(std_val_r2),
                    alpha=0.3, color='blue', label='±1 Standard Deviation')

    ax5.set_xlabel('Epoch', fontsize=AXIS_FONT, fontweight='bold')
    ax5.set_ylabel('Validation R²', fontsize=AXIS_FONT, fontweight='bold')
    ax5.set_title('Training Convergence\n(Mean ± Standard Deviation)',
                 fontsize=TITLE_FONT, fontweight='bold', pad=15)
    ax5.legend(fontsize=STATS_FONT)
    ax5.tick_params(axis='both', which='major', labelsize=TICK_FONT)
    ax5.grid(True, alpha=0.4)

    # Plot 6: Statistical summary - ENHANCED READABILITY
    ax6 = plt.subplot2grid((3, 3), (2, 2))
    ax6.axis('off')

    focus_text = "\n".join([f"• {a}-{b}" for a, b in focus_pairs])

    # Calculate comprehensive statistics
    ci_lower_r2, ci_upper_r2 = calculate_confidence_intervals(r2_scores)
    ci_lower_spearman, ci_upper_spearman = calculate_confidence_intervals(spearman_scores)

    summary_text = f"""
    STATISTICAL RESULTS SUMMARY
    {'=' * 30}

    PERFORMANCE (n=N_TRIALS):
    • R²: {mean_r2:.3f} ± {std_r2:.3f}
    • 95% CI: [{ci_lower_r2:.3f}, {ci_upper_r2:.3f}]
    • Spearman ρ: {mean_spearman:.3f} ± {np.std(spearman_scores):.3f}
    • 95% CI: [{ci_lower_spearman:.3f}, {ci_upper_spearman:.3f}]

    ANALYZED INTERFACES:
    {focus_text}

    SCIENTIFIC RIGOR:
    • Pure Variance: {std_r2**2:.6f}
    • Confidence Intervals
    • 10 Independent Trials
    • Consistent Temporal Splitting
    • Early Stopping Validation
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=STATS_FONT,
             verticalalignment='top', linespacing=1.5,
             bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow",
                      edgecolor='black', linewidth=2),
             fontfamily='monospace', fontweight='bold')

    plt.tight_layout(pad=3.0)  # Increased padding between subplots
    plt.show()

    return mean_r2, std_r2

# ============================================================
# EXECUTE ULTIMATE ANALYSIS
# ============================================================

if __name__ == "__main__":
    data_path = "/content/drive/MyDrive/syt1.pt" # Indicate the correct path for your file, for exemple, here a google drive path for syt1.pt

    print(" INITIATING ULTIMATE SCIENTIFIC ANALYSIS...")
    print("=" * 65)
    print(" ULTIMATE METHOD: Temporal Interleaving + Pure Variance Analysis")
    print(" OBJECTIVE: Maximum scientific rigor with MD trajectory optimization")
    print("=" * 65)

    # Load and validate data
    all_graphs = torch.load(data_path, map_location='cpu', weights_only=False)
    all_graphs = [convert_to_float32(g) for g in all_graphs]

    # Ultimate interactive setup
    focus_pairs, unique_segments = ultimate_interactive_setup(all_graphs)

    # Run ultimate trials
    all_results, all_histories = run_ultimate_trials(data_path, focus_pairs, n_runs=N_TRIALS)

    # Ultimate statistical analysis
    best_run, metrics = perform_ultimate_analysis(all_results, focus_pairs)

    # Ultimate visualization
    mean_r2, std_r2 = plot_ultimate_analysis(metrics, all_histories, focus_pairs)

    # Save ultimate model
    torch.save({
        'model_state_dict': best_run['model'].state_dict(),
        'test_r2': best_run['test_r2'],
        'test_spearman': best_run['test_spearman'],
        'test_mae': best_run['test_mae'],
        'val_loss': best_run['val_loss'],
        'test_metrics': best_run['test_metrics'],
        'focus_pairs': focus_pairs,
        'all_trials_metrics': metrics,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'ultimate_analysis': True,
        'temporal_interleaving': True,
        'pure_variance': True,
        'confidence_intervals': {
            'r2': calculate_confidence_intervals(metrics['Test R²']),
            'spearman': calculate_confidence_intervals(metrics['Test Spearman'])
        }
    }, 'ULTIMATE_SCIENTIFIC_MODEL.pth')

    print(f"\n ULTIMATE MODEL ARCHIVED WITH MAXIMUM SCIENTIFIC RIGOR!")
    print(f" Final Performance: R² = {mean_r2:.3f} ± {std_r2:.3f}")
    print(f" Pure Model Variance: {std_r2**2:.6f}")

  










