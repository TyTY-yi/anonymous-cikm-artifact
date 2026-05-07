"""
Heterogeneous Knowledge Graph Multi-Round Optimization
=======================================================
Implements the Multi-Round Graph Optimization described in Section 3.3 of the paper.
Three independent rounds with different random seeds are run; learned edge weights are averaged and pruned by threshold delta
to yield the task-optimal sparse graph G*.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from copy import deepcopy

# ================= CONFIGURATION =================
DATA_DIR = '<path/to/processed_data>'
KG_DIR = '<path/to/processed_data>'
OUTPUT_DIR = '<path/to/optimized_kg_results>'
VISUALIZATION_DIR = '<path/to/visualizations>'

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 1000
TEMP_START = 5.0
TEMP_END = 0.05

# Multi-round settings
NUM_ROUNDS = 3
RANDOM_SEEDS = [42, 123, 456]
MERGE_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]

# Loss weights (CRITICAL FIXES HERE)
WEIGHT_CLS = 1.0
WEIGHT_SPARSITY = 0.05
WEIGHT_KG_ALIGN = 0.1
WEIGHT_ENTROPY = 0.01

# Early stopping
PATIENCE = 50
MIN_DELTA = 0.001
VAL_RATIO = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================= DATA LOADING =================
def load_all_data():
    train_df = pd.read_csv(f'{DATA_DIR}/train_set.csv')
    train_emb = np.load(f'{DATA_DIR}/train_embeddings.npy')
    return train_df, train_emb


def load_node_embeddings():
    feature_emb = np.load(f'{KG_DIR}/feature_embeddings.npy')
    semantic_emb = np.load(f'{KG_DIR}/semantic_embeddings.npy')
    return feature_emb, semantic_emb


def load_initial_adjacency_matrices(round_idx=0):
    adj_dict = {
        'f2f_co_occurs': np.load(f'{KG_DIR}/adj_f2f_co_occurs.npy'),
        'f2f_precedes': np.load(f'{KG_DIR}/adj_f2f_precedes.npy'),
        'f2f_intensifies': np.load(f'{KG_DIR}/adj_f2f_intensifies.npy'),
        'f2f_contradicts': np.load(f'{KG_DIR}/adj_f2f_contradicts.npy'),
        'f2f_overlaps': np.load(f'{KG_DIR}/adj_f2f_overlaps.npy'),
        'f2s': np.load(f'{KG_DIR}/adj_f2s.npy'),
        's2l': np.load(f'{KG_DIR}/adj_s2l.npy'),
        'f2l': np.load(f'{KG_DIR}/adj_f2l.npy')
    }
    return adj_dict


def load_kg_metadata():
    with open(f'{KG_DIR}/kg_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


# ================= DATASET =================
class KGDataset(Dataset):
    def __init__(self, df, embeddings):
        self.embeddings = embeddings.astype(np.float32)
        if 'label_id' in df.columns:
            self.labels = df['label_id'].values
        elif 'label' in df.columns:
            if pd.api.types.is_numeric_dtype(df['label']):
                self.labels = df['label'].values
            else:
                label_map = {'no_reason': 0, 'bias_or_abuse': 1, 'jobs_and_careers': 2,
                             'medication': 3, 'relationship': 4, 'alienation': 5}
                self.labels = df['label'].map(label_map).values
        else:
            raise ValueError("Dataset must have either 'label' or 'label_id' column")
        self.texts = df['text'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.texts[idx]


# ================= MODEL =================
def gumbel_softmax(logits, temperature=1.0, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = logits + gumbel_noise
    y = torch.sigmoid(y / temperature)
    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y
    return y


class HeterogeneousKGOptimizer(nn.Module):
    def __init__(self, feature_emb, semantic_emb, initial_adj_dict):
        super().__init__()
        self.register_buffer('feature_emb', torch.FloatTensor(feature_emb))
        self.register_buffer('semantic_emb', torch.FloatTensor(semantic_emb))

        self.initial_adj = {}
        for edge_type, adj in initial_adj_dict.items():
            self.initial_adj[edge_type] = torch.FloatTensor(adj)

        self.edge_logits = nn.ParameterDict()
        for edge_type, adj in initial_adj_dict.items():
            adj_tensor = torch.FloatTensor(adj)
            logits = torch.where(adj_tensor > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
            self.edge_logits[edge_type] = nn.Parameter(logits)

        self.classifier = nn.Linear(6, 6)  # 6 labels
        self.statistics = {}

    def set_statistics(self, statistics):
        self.statistics = statistics

    def normalize_propagation(self, features, mask):
        """
        Mean Aggregation: Normalize by degree to prevent value explosion
        features: [B, N_in]
        mask: [N_in, N_out]
        """
        # Degree of destination nodes + epsilon to avoid div by zero
        degree = mask.sum(dim=0, keepdim=True).clamp(min=1.0)
        output = features @ mask
        return output / degree

    def forward(self, text_embedding, temperature=1.0, training=True):
        batch_size = text_embedding.shape[0]

        # Activate features: Dot product similarity
        feature_scores = text_embedding @ self.feature_emb.T  # [B, N_feature]

        # Get Edge Masks
        masks = {}
        for k, v in self.edge_logits.items():
            if training:
                masks[k] = gumbel_softmax(v, temperature, hard=False)
            else:
                masks[k] = torch.sigmoid(v)

        # Feature-to-Feature Propagation (Aggregated Mean)
        f2f_combined = (
                masks['f2f_co_occurs'] +
                masks['f2f_precedes'] +
                masks['f2f_intensifies'] +
                masks['f2f_overlaps'] -
                (masks['f2f_contradicts'] * 0.8)
        )
        f2f_combined = F.relu(f2f_combined)

        # Apply propagation with Mean Aggregation
        f2f_influence = self.normalize_propagation(feature_scores, f2f_combined)

        # Residual connection + Influence
        feature_scores = feature_scores + f2f_influence
        feature_scores = F.layer_norm(feature_scores, feature_scores.shape)  # Normalization helps stability

        # 4. Feature -> Semantic Propagation
        semantic_scores = self.normalize_propagation(feature_scores, masks['f2s'])  # [B, N_cat]

        # 5. Semantic -> Label Propagation
        label_from_semantics = self.normalize_propagation(semantic_scores, masks['s2l'])  # [B, N_label]

        # 6. Feature -> Label Propagation
        label_from_features = self.normalize_propagation(feature_scores, masks['f2l'])  # [B, N_label]

        label_logits = (label_from_semantics + label_from_features) / 2

        # Final classification
        out = self.classifier(label_logits)

        return out, masks


# ================= LOSS FUNCTION =================
def compute_heterogeneous_loss(model, logits, labels, masks, statistics):
    """
    Revised Loss Function:
    1. Cross Entropy (Task Loss)
    2. Sparsity (L1 regularization on masks)
    3. Entropy (Push weights to 0 or 1)
    4. KG Alignment (Weak constraint)
    """

    # Task Loss
    loss_cls = F.cross_entropy(logits, labels)

    # Sparsity Loss
    loss_sparsity = 0
    total_elements = 0
    for mask in masks.values():
        loss_sparsity += torch.mean(mask)
        total_elements += 1
    loss_sparsity = loss_sparsity / total_elements

    # Entropy Loss
    loss_entropy = 0
    for mask in masks.values():
        # Clamp to avoid log(0)
        p = torch.clamp(mask, 1e-6, 1 - 1e-6)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        loss_entropy += torch.mean(entropy)
    loss_entropy = loss_entropy / total_elements

    # KG Alignment
    loss_kg = 0
    count = 0
    for edge_type, mask in masks.items():
        if edge_type in model.initial_adj:
            initial = model.initial_adj[edge_type].to(mask.device)
            loss_kg += F.binary_cross_entropy(mask, initial)
            count += 1
    loss_kg = loss_kg / count if count > 0 else 0

    # Total Loss
    total_loss = (
            WEIGHT_CLS * loss_cls +
            WEIGHT_SPARSITY * loss_sparsity +
            WEIGHT_KG_ALIGN * loss_kg +
            WEIGHT_ENTROPY * loss_entropy
    )

    return total_loss, {
        'total': total_loss.item(),
        'cls': loss_cls.item(),
        'sparsity': loss_sparsity.item(),
        'kg_align': loss_kg.item(),
        'entropy': loss_entropy.item()
    }


# ================= TRAINING =================
def train_one_round(round_idx, train_df, train_emb, feature_emb, semantic_emb,
                    initial_adj_dict, statistics, metadata):
    print("\n" + "=" * 70)
    print(f"ROUND {round_idx + 1}/{NUM_ROUNDS} (Seed: {RANDOM_SEEDS[round_idx]})")
    print("=" * 70)

    set_seed(RANDOM_SEEDS[round_idx])
    device = get_device()

    # Split train/val
    val_size = int(len(train_df) * VAL_RATIO)
    train_size = len(train_df) - val_size

    # Shuffle indices before split to ensure randomness per seed
    indices = np.random.permutation(len(train_df))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset_df = train_df.iloc[train_indices].reset_index(drop=True)
    val_subset_df = train_df.iloc[val_indices].reset_index(drop=True)
    train_subset_emb = train_emb[train_indices]
    val_subset_emb = train_emb[val_indices]

    train_dataset = KGDataset(train_subset_df, train_subset_emb)
    val_dataset = KGDataset(val_subset_df, val_subset_emb)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HeterogeneousKGOptimizer(feature_emb, semantic_emb, initial_adj_dict)
    model.set_statistics(statistics)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_f1': []}

    for epoch in range(MAX_EPOCHS):
        progress = epoch / MAX_EPOCHS
        temperature = max(TEMP_END, TEMP_START * (0.99 ** epoch))  # Exponential decay

        # Train
        model.train()
        train_loss_accum = 0

        for batch_emb, batch_labels, _ in train_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits, masks = model(batch_emb, temperature=temperature, training=True)
            loss, loss_dict = compute_heterogeneous_loss(model, logits, batch_labels, masks, statistics)

            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation (Calculate F1)
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_emb, batch_labels, _ in val_loader:
                batch_emb = batch_emb.to(device)
                batch_labels = batch_labels.to(device)

                logits, masks = model(batch_emb, temperature=temperature, training=False)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)

        history['train_loss'].append(avg_train_loss)
        history['val_f1'].append(val_f1)

        # Debug Edge Weights
        if epoch % 10 == 0:
            with torch.no_grad():
                sample_mask = torch.sigmoid(model.edge_logits['f2f_co_occurs']).mean()
                print(
                    f"  Epoch {epoch:3d} | Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f} | Mean Edge Prob: {sample_mask:.4f}")

        # Early stopping on F1
        if val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"  Best Val F1: {best_val_f1:.4f}")

    # Extract optimized adjacency
    model.eval()
    optimized_adj = {}
    with torch.no_grad():
        for edge_type, logits in model.edge_logits.items():
            probs = torch.sigmoid(logits).cpu().numpy()
            optimized_adj[edge_type] = probs

            # Count retained edges for logging
            retained_count = (probs > 0.5).sum()
            print(f"    {edge_type}: retained {retained_count}/{probs.size} edges (kept probs)")

    return optimized_adj, history, best_val_f1


# ================= POST-TRAINING PIPELINE =================

def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def merge_round_results(round_results):
    """Merge adjacency matrices from multiple rounds using weighted average"""
    print("\n" + "=" * 70)
    print("MERGING ROUND RESULTS")
    print("=" * 70)

    merged_adj = {}
    edge_types = round_results[0].keys()

    for edge_type in edge_types:
        # Weighted average
        merged = np.zeros_like(round_results[0][edge_type])
        for round_idx, adj_dict in enumerate(round_results):
            weight = MERGE_WEIGHTS[round_idx]
            merged += weight * adj_dict[edge_type]

        merged_adj[edge_type] = merged
        n_edges_before = sum([(adj > 0.5).sum() for adj in [r[edge_type] for r in round_results]]) / NUM_ROUNDS
        n_edges_after = (merged > 0.5).sum()
        print(f"  {edge_type:20s}: avg {n_edges_before:.0f} -> {n_edges_after:.0f} edges (threshold > 0.5)")

    return merged_adj



def create_weighted_kg_json(original_kg, optimized_adj, metadata):
    """Create weighted KG JSON with optimized edge probabilities."""
    kg = deepcopy(original_kg)

    # Update metadata
    kg['metadata']['optimization_method'] = '3-round heterogeneous KG optimization (Fixed F1 & Mean Agg)'
    kg['metadata']['edge_weights'] = 'Optimized probabilities (0.0-1.0)'

    # Load node index mappings
    with open(f'{KG_DIR}/node_index_mappings.json', 'r') as f:
        mappings = json.load(f)

    # Update edge weights
    for edge in kg['edges']:
        edge_type = edge.get('edge_type')
        source = edge.get('source_id') or edge.get('source')
        target = edge.get('target_id') or edge.get('target')
        subtype = edge.get('subtype', '')

        weight = 0.0

        # Look up corresponding adjacency matrix by edge type
        matrix_key = None
        if edge_type == 'feature_to_feature':
            if subtype == 'co-occurs-in':
                matrix_key = 'f2f_co_occurs'
            elif subtype == 'precedes':
                matrix_key = 'f2f_precedes'
            elif subtype == 'intensifies':
                matrix_key = 'f2f_intensifies'
            elif subtype == 'contradicts':
                matrix_key = 'f2f_contradicts'
            elif subtype == 'overlaps-with':
                matrix_key = 'f2f_overlaps'
        elif edge_type == 'feature_to_semantic':
            matrix_key = 'f2s'
        elif edge_type == 'semantic_to_label':
            matrix_key = 's2l'
        elif edge_type == 'feature_to_label':
            matrix_key = 'f2l'

        # Retrieve weight value
        if matrix_key and matrix_key in optimized_adj:
            src_map = mappings['feature_id_to_idx'] if 'feature' in edge_type.split('_')[0] else mappings[
                'semantic_id_to_idx']
            tgt_map = mappings['label_id_to_idx'] if 'label' in edge_type.split('_')[-1] else (
                mappings['semantic_id_to_idx'] if 'semantic' in edge_type.split('_')[-1] else mappings[
                    'feature_id_to_idx'])

            src_idx = src_map.get(source)
            tgt_idx = tgt_map.get(target)

            if src_idx is not None and tgt_idx is not None:
                # Retrieve edge probability as weight
                weight = float(optimized_adj[matrix_key][src_idx, tgt_idx])

        edge['weight'] = weight
        edge['optimized'] = True

    return kg



def save_results(merged_adj, round_results, histories, metadata):
    """Save all results: optimized numpy matrices and JSON knowledge graph files."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # 1. Save numpy matrices
    for edge_type, adj in merged_adj.items():
        np.save(f'{OUTPUT_DIR}/adj_{edge_type}_optimized.npy', adj)
        print(f"  Saved adj_{edge_type}_optimized.npy")

    # 2. Save training histories
    with open(f'{OUTPUT_DIR}/training_histories.json', 'w') as f:
        json.dump(convert_to_json_serializable(histories), f, indent=2)

    # 3. Generate and save JSON knowledge graph files
    print("\n  Generating JSON knowledge graph files...")

    kg_path = '<path/to/kg.json>'
    if os.path.exists(kg_path):
        with open(kg_path, 'r', encoding='utf-8') as f:
            original_kg = json.load(f)

        # Save weighted version
        weighted_kg = create_weighted_kg_json(original_kg, merged_adj, metadata)
        with open(f'{OUTPUT_DIR}/optimized_kg_weighted.json', 'w', encoding='utf-8') as f:
            json.dump(weighted_kg, f, indent=2, ensure_ascii=False)
        print(f"  Saved optimized_kg_weighted.json")


    else:
        print(f"  Warning: {kg_path} not found. Cannot generate JSON files.")

    print(f"\n{'=' * 70}")
    print(f"ALL RESULTS SAVED TO: {OUTPUT_DIR}/")
    print(f"{'=' * 70}")


def main():
    # Load data
    train_df, train_emb = load_all_data()
    feature_emb, semantic_emb = load_node_embeddings()
    metadata = load_kg_metadata()


    statistics = {}

    round_results = []
    histories = []

    for round_idx in range(NUM_ROUNDS):
        initial_adj_dict = load_initial_adjacency_matrices(round_idx)
        optimized_adj, history, best_f1 = train_one_round(
            round_idx, train_df, train_emb, feature_emb, semantic_emb,
            initial_adj_dict, statistics, metadata
        )
        round_results.append(optimized_adj)
        histories.append(history)

    merged_adj = merge_round_results(round_results)
    save_results(merged_adj, round_results, histories, metadata)


if __name__ == "__main__":
    main()