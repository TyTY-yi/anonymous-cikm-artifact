"""
Knowledge Graph Encoding Script
=================================
Encodes all node types (feature, semantic, label) using MentalBERT and
builds adjacency matrices for all edge types. Outputs are saved to OUTPUT_DIR
for use by the graph optimization and inference stages.
"""

import json
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import defaultdict

# ================= CONFIGURATION =================
# Path to the input knowledge graph JSON file
INPUT_KG_PATH = '<path/to/kg.json>'
# Directory where encoded outputs will be saved
OUTPUT_DIR = '<path/to/output/directory>'
MODEL_NAME = 'mental/mental-bert-base-uncased'

# Label order
LABEL_ORDER = [
    'no_reason',  # Label 0
    'bias_or_abuse',  # Label 1
    'jobs_and_careers',  # Label 2
    'medication',  # Label 3
    'relationship',  # Label 4
    'alienation'  # Label 5
]

# Feature-to-feature edge subtypes
F2F_SUBTYPES = [
    'co-occurs-in',
    'precedes',
    'intensifies',
    'contradicts',
    'overlaps-with'
]


# ===============================================
def get_device():
    """Get available device (CUDA or CPU)"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def load_knowledge_graph(path):
    """Load the knowledge graph JSON"""
    print(f"Loading knowledge graph from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        kg = json.load(f)

    print(f"  Loaded KG with:")
    print(f"    - {len(kg['label_nodes'])} label nodes (Layer 3)")
    print(f"    - {len(kg['semantic_nodes'])} semantic nodes (Layer 2)")
    print(f"    - {len(kg['feature_nodes'])} feature nodes (Layer 1)")
    print(f"    - {len(kg['edges'])} total edges")

    return kg

def find_node_by_id(node_list, node_id):
    """Helper function to find a node by its ID"""
    for node in node_list:
        if (node.get('feature_id') == node_id or
                node.get('semantic_id') == node_id or
                node.get('label_id') == node_id):
            return node
    return None

def construct_feature_text(feature_node, kg):
    """
    Construct rich text representation for a feature node

    Includes:
    - Feature name and definition
    - Informal expressions
    - Valence, severity, feature type
    - Primary semantic
    - Observable status

    Excludes (as requested):
    - theoretical_basis
    """
    parts = []
    feature_id = feature_node['feature_id']

    # Core information
    parts.append(f"Attribution feature: {feature_node['feature_name']}")
    parts.append(f"Definition: {feature_node['definition']}")

    # Metadata
    parts.append(f"Feature type: {feature_node['feature_type']}")
    parts.append(f"Valence: {feature_node['valence']}")
    parts.append(f"Severity: {feature_node['severity']}")

    # Observable status
    observable_str = "yes" if feature_node['observable'] else "no"
    parts.append(f"Observable: {observable_str}")

    # Primary semantic
    parts.append(f"Primary semantic: {feature_node['primary_semantic']}")

    # Informal expressions
    if feature_node.get('informal_expressions'):
        informal = feature_node['informal_expressions']
        informal_text = ", ".join(informal)
        parts.append(f"Common patient expressions: {informal_text}")

    # Category connections (feature_to_semantic edges)
    semantic_contexts = []
    for edge in kg['edges']:
        if edge.get('edge_type') == 'feature_to_semantic':
            # Handle both 'source'/'target' and 'source_id'/'target_id' formats
            source = edge.get('source_id') or edge.get('source')
            target = edge.get('target_id') or edge.get('target')

            if source == feature_id:
                semantic_node = find_node_by_id(kg['semantic_nodes'], target)
                if semantic_node:
                    semantic_contexts.append(semantic_node['semantic_name'])

    if semantic_contexts:
        parts.append(f"Connected to semantic nodes: {', '.join(semantic_contexts)}")

    return ". ".join(parts)


def construct_semantic_text(semantic_node):
    parts = []

    parts.append(f"Semantic node: {semantic_node['semantic_name']}")
    parts.append(f"Definition: {semantic_node['definition']}")
    parts.append(f"Primary label: {semantic_node['primary_label']}")

    # Secondary labels
    if semantic_node.get('secondary_labels'):
        secondary = ", ".join(semantic_node['secondary_labels'])
        parts.append(f"Secondary labels: {secondary}")

    # Mediates
    if semantic_node.get('mediates'):
        parts.append(f"Mediates: {semantic_node['mediates']}")

    return ". ".join(parts)


def construct_label_text(label_node):
    parts = []

    parts.append(f"Mental health attribution label: {label_node['label_name']}")
    parts.append(f"Definition: {label_node['definition']}")
    parts.append(f"Characteristics: {label_node['characteristics']}")
    parts.append(f"Clinical significance: {label_node['clinical_significance']}")

    return ". ".join(parts)


def build_index_mappings(kg):
    """
    Build mappings from node IDs to array indices
    Returns: feature_map, semantic_map, label_map
    """
    print("\n" + "=" * 70)
    print("BUILDING INDEX MAPPINGS")
    print("=" * 70)

    # Feature nodes (Layer 1)
    feature_map = {node['feature_id']: idx
                   for idx, node in enumerate(kg['feature_nodes'])}
    # Semantic nodes (Layer 2)
    semantic_map = {node['semantic_id']: idx
                    for idx, node in enumerate(kg['semantic_nodes'])}
    # Label nodes (Layer 3)
    label_map = {}
    for node in kg['label_nodes']:
        label_name = node['label_name']
        if label_name in LABEL_ORDER:
            label_map[node['label_id']] = LABEL_ORDER.index(label_name)

    print(f"  Feature ID to index: {len(feature_map)} mappings")
    print(f"  Semantic ID to index: {len(semantic_map)} mappings")
    print(f"  Label ID to index: {len(label_map)} mappings")

    return feature_map, semantic_map, label_map


def encode_nodes(kg, tokenizer, model, device):
    """
    Encode all node types using Mental-BERT
    Returns embeddings for features, semantics, and labels
    """
    print("\n" + "=" * 70)
    print("ENCODING NODE EMBEDDINGS")
    print("=" * 70)

    model.eval()

    # ========== Encode Features (Layer 1) ==========
    print("\n Encoding feature nodes (Layer 1)...")
    feature_embeddings = []

    for node in tqdm(kg['feature_nodes'], desc="  Features"):
        text = construct_feature_text(node, kg)

        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            feature_embeddings.append(cls_embedding)

    feature_matrix = np.array(feature_embeddings)
    print(f"  Feature embeddings shape: {feature_matrix.shape}")

    # ========== Encode Categories (Layer 2) ==========
    print("\n Encoding semantic nodes (Layer 2)...")
    semantic_embeddings = []

    for node in tqdm(kg['semantic_nodes'], desc="  Semantics"):
        text = construct_semantic_text(node)

        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            semantic_embeddings.append(cls_embedding)

    semantic_matrix = np.array(semantic_embeddings)
    print(f"  Category embeddings shape: {semantic_matrix.shape}")

    # ========== Encode Labels (Layer 3) ==========
    print("\n Encoding label nodes (Layer 3)...")
    label_embeddings = []

    # Sort labels by predefined order
    sorted_labels = sorted(kg['label_nodes'],
                           key=lambda x: LABEL_ORDER.index(x['label_name']))

    for node in tqdm(sorted_labels, desc="  Labels"):
        text = construct_label_text(node)

        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            label_embeddings.append(cls_embedding)

    label_matrix = np.array(label_embeddings)
    print(f"  Label embeddings shape: {label_matrix.shape}")

    return {
        'feature': feature_matrix,
        'semantic': semantic_matrix,
        'label': label_matrix
    }


def build_adjacency_matrices(kg, feature_map, semantic_map, label_map):
    """
    Build adjacency matrices for all edge types
    """
    print("\n" + "=" * 70)
    print("BUILDING ADJACENCY MATRICES")
    print("=" * 70)

    n_features = len(feature_map)
    n_semantics = len(semantic_map)
    n_labels = len(label_map)

    # Initialize all adjacency matrices
    adj_matrices = {
        'f2s': np.zeros((n_features, n_semantics), dtype=np.float32),
        's2l': np.zeros((n_semantics, n_labels), dtype=np.float32),
        'f2l': np.zeros((n_features, n_labels), dtype=np.float32),
        'f2f_co_occurs': np.zeros((n_features, n_features), dtype=np.float32),
        'f2f_precedes': np.zeros((n_features, n_features), dtype=np.float32),
        'f2f_intensifies': np.zeros((n_features, n_features), dtype=np.float32),
        'f2f_contradicts': np.zeros((n_features, n_features), dtype=np.float32),
        'f2f_overlaps': np.zeros((n_features, n_features), dtype=np.float32)
    }

    edge_counts = defaultdict(int)
    skipped_edges = defaultdict(int)

    print("\n  Processing edges...")

    for edge in tqdm(kg['edges'], desc="  Edges"):
        edge_type = edge['edge_type']

        # Handle both 'source'/'target' and 'source_id'/'target_id' formats
        source_id = edge.get('source_id') or edge.get('source')
        target_id = edge.get('target_id') or edge.get('target')

        # ========== Feature-to-Semantic ==========
        if edge_type == 'feature_to_semantic':
            if source_id not in feature_map or target_id not in semantic_map:
                skipped_edges['f2s'] += 1
                continue

            src_idx = feature_map[source_id]
            tgt_idx = semantic_map[target_id]
            adj_matrices['f2s'][src_idx, tgt_idx] = 1
            edge_counts['f2s'] += 1

        # ========== Semantic-to-Label ==========
        elif edge_type == 'semantic_to_label':
            if source_id not in semantic_map or target_id not in label_map:
                skipped_edges['s2l'] += 1
                continue

            src_idx = semantic_map[source_id]
            tgt_idx = label_map[target_id]
            adj_matrices['s2l'][src_idx, tgt_idx] = 1
            edge_counts['s2l'] += 1

        # ========== Feature-to-Label (Direct) ==========
        elif edge_type == 'feature_to_label':
            if source_id not in feature_map or target_id not in label_map:
                skipped_edges['f2l'] += 1
                continue

            src_idx = feature_map[source_id]
            tgt_idx = label_map[target_id]
            adj_matrices['f2l'][src_idx, tgt_idx] = 1
            edge_counts['f2l'] += 1

        # ========== Feature-to-Feature (5 subtypes) ==========
        elif edge_type == 'feature_to_feature':
            if source_id not in feature_map or target_id not in feature_map:
                skipped_edges['f2f'] += 1
                continue

            src_idx = feature_map[source_id]
            tgt_idx = feature_map[target_id]
            subtype = edge.get('subtype', '')

            # Co-occurs (undirected)
            if subtype == 'co-occurs-in':
                adj_matrices['f2f_co_occurs'][src_idx, tgt_idx] = 1
                adj_matrices['f2f_co_occurs'][tgt_idx, src_idx] = 1
                edge_counts['f2f_co_occurs'] += 1

            # Precedes (directed)
            elif subtype == 'precedes':
                adj_matrices['f2f_precedes'][src_idx, tgt_idx] = 1
                edge_counts['f2f_precedes'] += 1

            # Intensifies (directed)
            elif subtype == 'intensifies':
                adj_matrices['f2f_intensifies'][src_idx, tgt_idx] = 1
                edge_counts['f2f_intensifies'] += 1

            # Contradicts (undirected)
            elif subtype == 'contradicts':
                adj_matrices['f2f_contradicts'][src_idx, tgt_idx] = 1
                adj_matrices['f2f_contradicts'][tgt_idx, src_idx] = 1
                edge_counts['f2f_contradicts'] += 1

            # Overlaps (undirected)
            elif subtype == 'overlaps-with':
                adj_matrices['f2f_overlaps'][src_idx, tgt_idx] = 1
                adj_matrices['f2f_overlaps'][tgt_idx, src_idx] = 1
                edge_counts['f2f_overlaps'] += 1

    # Print statistics
    print("\n  Edge Statistics:")
    for matrix_name in sorted(adj_matrices.keys()):
        count = edge_counts.get(matrix_name, 0)
        print(f"    {matrix_name:20s}: {count:3d} edges")

    if skipped_edges:
        print("\n  Skipped edges (ID mismatch):")
        for matrix_name, count in sorted(skipped_edges.items()):
            print(f"    {matrix_name:20s}: {count:3d} edges")

    return adj_matrices, edge_counts


def save_all_outputs(embeddings, adj_matrices, kg,
                     feature_map, semantic_map, label_map,
                     edge_counts, output_dir):
    """
    Save all embeddings, adjacency matrices, and metadata to output directory
    """
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ========== Save Node Embeddings ==========
    print("\n Saving node embeddings...")
    np.save(os.path.join(output_dir, 'feature_embeddings.npy'), embeddings['feature'])
    print(f"  feature_embeddings.npy: {embeddings['feature'].shape}")

    np.save(os.path.join(output_dir, 'semantic_embeddings.npy'), embeddings['semantic'])
    print(f"  semantic_embeddings.npy: {embeddings['semantic'].shape}")

    np.save(os.path.join(output_dir, 'label_embeddings.npy'), embeddings['label'])
    print(f"  label_embeddings.npy: {embeddings['label'].shape}")

    # ========== Save Adjacency Matrices ==========
    print("\n Saving adjacency matrices...")
    for matrix_name, matrix in adj_matrices.items():
        filename = f'adj_{matrix_name}.npy'
        np.save(os.path.join(output_dir, filename), matrix)
        print(f"  {filename:30s}: {matrix.shape}")

    # ========== Save Metadata ==========
    print("\n Saving metadata...")

    # Extract node names
    feature_names = [node['feature_name'] for node in kg['feature_nodes']]
    semantic_names = [node['semantic_name'] for node in kg['semantic_nodes']]
    label_names = LABEL_ORDER

    # Extract feature features
    feature_types = [node['feature_type'] for node in kg['feature_nodes']]
    feature_valences = [node['valence'] for node in kg['feature_nodes']]
    feature_severities = [node['severity'] for node in kg['feature_nodes']]
    feature_observable = [node['observable'] for node in kg['feature_nodes']]
    feature_primary_cats = [node['primary_semantic'] for node in kg['feature_nodes']]

    metadata = {
        "graph_name": "Attribution_Knowledge_Graph",
        "version": "1.0_encoded",

        "dataset": "dataset",

        "node_counts": {
            "features": len(feature_map),
            "semantics": len(semantic_map),
            "labels": len(label_map)
        },

        "node_orders": {
            "feature_order": feature_names,
            "semantic_order": semantic_names,
            "label_order": label_names
        },

        "feature_metadata": {
            "feature_types": feature_types,
            "valences": feature_valences,
            "severities": feature_severities,
            "observable": feature_observable,
            "primary_semantics": feature_primary_cats
        },

        "edge_statistics": {
            matrix_name: {
                "count": int(edge_counts.get(matrix_name, 0)),
                "shape": list(adj_matrices[matrix_name].shape),
                "directed": matrix_name in ['f2f_precedes', 'f2f_intensifies']
            }
            for matrix_name in adj_matrices.keys()
        },

        "total_edges": sum(edge_counts.values()),

        "adjacency_matrix_files": list(adj_matrices.keys()),

        "embedding_files": [
            "feature_embeddings.npy",
            "semantic_embeddings.npy",
            "label_embeddings.npy"
        ],

        "edge_types": {
            "f2s": "feature_to_semantic (Layer 1 → Layer 2)",
            "s2l": "semantic_to_label (Layer 2 → Layer 3)",
            "f2l": "feature_to_label (Layer 1 → Layer 3, direct)",
            "f2f_co_occurs": "feature co-occurs with feature (undirected)",
            "f2f_precedes": "feature precedes feature (directed)",
            "f2f_intensifies": "feature intensifies feature (directed)",
            "f2f_contradicts": "feature contradicts feature (undirected)",
            "f2f_overlaps": "feature overlaps with feature (undirected)"
        },

        "notes": [
            "theoretical_basis was excluded from encoding as requested",
            "All embeddings generated using mental/mental-bert-base-uncased",
            "All informal_expressions are included in feature encoding",
            "Undirected edges have symmetric adjacency matrices",
            "Label order: no_reason(0), bias_or_abuse(1), jobs_and_careers(2), medication(3), relationship(4), alienation(5)"
        ]
    }

    metadata_path = os.path.join(output_dir, 'kg_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  kg_metadata.json")

    # ========== Save Index Mappings ==========
    print("\n Saving index mappings...")
    mappings = {
        "feature_id_to_idx": feature_map,
        "semantic_id_to_idx": semantic_map,
        "label_id_to_idx": label_map
    }

    mappings_path = os.path.join(output_dir, 'node_index_mappings.json')
    with open(mappings_path, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)

    print(f"  node_index_mappings.json")

    print(f"\n{'=' * 70}")
    print(f"ALL OUTPUTS SAVED TO: {output_dir}/")
    print(f"{'=' * 70}")


def main():
    """Main execution pipeline"""
    print(f"Device: {get_device()}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Load knowledge graph
    kg = load_knowledge_graph(INPUT_KG_PATH)

    # 2. Load Mental-BERT model
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"  Model loaded on {device}")

    # 3. Build index mappings
    feature_map, semantic_map, label_map = build_index_mappings(kg)

    # 4. Encode all nodes
    embeddings = encode_nodes(kg, tokenizer, model, device)

    # 5. Build adjacency matrices
    adj_matrices, edge_counts = build_adjacency_matrices(
        kg, feature_map, semantic_map, label_map
    )

    # 6. Save everything
    save_all_outputs(
        embeddings, adj_matrices, kg,
        feature_map, semantic_map, label_map,
        edge_counts, OUTPUT_DIR
    )


if __name__ == "__main__":
    main()