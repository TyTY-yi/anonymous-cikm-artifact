"""
Evidence-Based Subgraph Construction and LLM-Assisted Classification
====================================================================================
Implements the Section 3.4 and 3.5 of the paper.

Purpose:
Construct sparse evidence subgraphs from input texts using the optimized heterogeneous KG
Use n-hop neighborhood expansion with confidence thresholding
Augment LLM reasoning with retrieved semantically similar examples via RAG
Save subgraph structures in classification results

Key Features:
Feature detection from text using informal expressions
Multi-hop expansion:
  * Hop 1: features -> features
  * Hop 2: features -> semantic, semantic -> labels, features -> labels
Edge pruning based on optimized weights
Preserve subgraph structure in results
"""

import ollama
import json
import numpy as np
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
DATA_DIR = '<path/to/processed_data>'
OPTIMIZED_KG_DIR = '<path/to/optimized_kg_results>'
BASE_KG_PATH = '<path/to/kg.json>'

# Model configuration
MODEL_NAME = " "
USE_THINKING_MODE = False
OLLAMA_HOST = ' '

# Subgraph construction parameters
FEATURE_DETECTION_THRESHOLD = 0.55
EDGE_WEIGHT_THRESHOLD = 0.3
MAX_FEATURES_PER_TEXT = 10
N_HOP_EXPANSION = 2
MAX_EDGES_PER_SUBGRAPH = 35

# RAG parameters
TOP_K_TEXT = 5


# Label mappings
ID_TO_LABEL = {
    0: 'no_reason',
    1: 'bias_or_abuse',
    2: 'jobs_and_careers',
    3: 'medication',
    4: 'relationship',
    5: 'alienation'
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle other types
        if hasattr(obj, 'item'):
            return obj.item()
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        # Handle any other object by converting to string
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


# ================= LOAD KNOWLEDGE GRAPHS =================
def load_base_kg():
    """Load the original base KG with feature definitions and informal expressions"""
    with open(BASE_KG_PATH, 'r', encoding='utf-8') as f:
        kg = json.load(f)
    print(f"Loaded base KG: {len(kg['feature_nodes'])} features, "
          f"{len(kg['semantic_nodes'])} semantic nodes, {len(kg['label_nodes'])} labels")
    return kg


def load_optimized_kg():
    """Load the optimized heterogeneous KG (weighted edges)"""
    with open(f'{OPTIMIZED_KG_DIR}/optimized_kg_weighted.json', 'r', encoding='utf-8') as f:
        opt_kg = json.load(f)
    print(f"Loaded optimized KG: {len(opt_kg['edges'])} edges")
    return opt_kg


# ================= FEATURE DETECTION =================
class FeatureDetector:
    """Detect attribution features in text using informal expressions from base KG"""

    def __init__(self, base_kg):
        self.feature_patterns = self._build_patterns(base_kg)
        self.feature_id_to_name = {
            node['feature_id']: node['feature_name']
            for node in base_kg['feature_nodes']
        }
        self.feature_name_to_id = {v: k for k, v in self.feature_id_to_name.items()}

    def _build_patterns(self, kg):
        """Build regex patterns for each feature"""
        patterns = {}

        for node in kg['feature_nodes']:
            feature_id = node['feature_id']
            expressions = node.get('informal_expressions', [])

            # Filter out very short expressions (< 3 chars) to reduce false positives
            expressions = [expr for expr in expressions if len(expr) >= 3]

            patterns[feature_id] = expressions

        return patterns

    def detect_features(self, text):
        """Detect features in text with confidence scores"""
        text_lower = text.lower()
        detected = []

        for feature_id, expressions in self.feature_patterns.items():
            matched_exprs = []

            for expr in expressions:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(expr) + r'\b'
                if re.search(pattern, text_lower):
                    matched_exprs.append(expr)

            if matched_exprs:
                # Confidence based on number of matched expressions
                confidence = min(1.0, len(matched_exprs) / 3.0)

                detected.append({
                    'feature_id': feature_id,
                    'feature_name': self.feature_id_to_name[feature_id],
                    'confidence': confidence,
                    'matched_expressions': matched_exprs
                })

        # Sort by confidence
        detected.sort(key=lambda x: x['confidence'], reverse=True)

        return detected

# ================= EVIDENCE SUBGRAPH BUILDER =================
class EvidenceSubgraphBuilder:
    """
    Build sparse evidence subgraphs for mental health text classification
    Uses n-hop expansion in heterogeneous graph
    """

    def __init__(self, base_kg, optimized_kg):
        self.base_kg = base_kg
        self.opt_kg = optimized_kg

        # Build edge indices for efficient lookup
        self._build_edge_indices()

        # Build node mappings
        self.feature_nodes = {n['feature_id']: n for n in base_kg['feature_nodes']}
        self.semantic_nodes = {n['semantic_id']: n for n in base_kg['semantic_nodes']}
        self.label_nodes = {n['label_id']: n for n in base_kg['label_nodes']}

    def _build_edge_indices(self):
        """Build indices for fast edge lookup"""
        self.edge_index = {
            'feature_to_semantic': {},
            'semantic_to_label': {},
            'feature_to_feature': {},
            'feature_to_label': {}
        }

        # Process each edge (edges is a list)
        for edge in self.opt_kg['edges']:
            edge_type = edge['edge_type']
            source = edge.get('source_id') or edge.get('source')
            target = edge.get('target_id') or edge.get('target')
            weight = edge.get('weight', 0.0)

            if edge_type == 'feature_to_semantic':
                if source not in self.edge_index[edge_type]:
                    self.edge_index[edge_type][source] = []
                self.edge_index[edge_type][source].append((target, weight))

            elif edge_type == 'semantic_to_label':
                if source not in self.edge_index[edge_type]:
                    self.edge_index[edge_type][source] = []
                self.edge_index[edge_type][source].append((target, weight))

            elif edge_type == 'feature_to_feature':
                subtype = edge.get('subtype', 'unknown')
                if source not in self.edge_index[edge_type]:
                    self.edge_index[edge_type][source] = []
                self.edge_index[edge_type][source].append((target, weight, subtype))

            elif edge_type == 'feature_to_label':
                if source not in self.edge_index[edge_type]:
                    self.edge_index[edge_type][source] = []
                self.edge_index[edge_type][source].append((target, weight))

    def build_subgraph(self, detected_features, max_features=MAX_FEATURES_PER_TEXT,
                       edge_threshold=EDGE_WEIGHT_THRESHOLD, n_hops=N_HOP_EXPANSION,
                       max_edges=MAX_EDGES_PER_SUBGRAPH):
        """
        Build evidence subgraph from detected features using n-hop expansion
        Multi-hop Strategy:
        - Hop 1: Expand feature -> feature connections
        - Hop 2: Expand to higher layers:
          * Feature -> Semantic
          * Semantic -> Label
          * Feature -> Label

        Args:
            detected_features: List of detected features with confidence scores
            max_features: Maximum number of features to include
            edge_threshold: Minimum edge weight to include
            n_hops: Number of hops to expand
            max_edges: Maximum edges in subgraph

        Returns:
            Dictionary containing subgraph structure
        """

        # Filter and limit features
        detected_features = [f for f in detected_features if f['confidence'] >= FEATURE_DETECTION_THRESHOLD]
        detected_features = detected_features[:max_features]

        if not detected_features:
            return {
                'nodes': {'features': [], 'semantics': [], 'labels': []},
                'edges': [],
                'statistics': {'n_features': 0, 'n_semantics': 0, 'n_labels': 0, 'n_edges': 0, 'detected_features': 0}
            }

        # Initialize subgraph
        subgraph_nodes = {
            'features': set(),
            'semantics': set(),
            'labels': set()
        }
        subgraph_edges = []

        # Add detected features
        for feat in detected_features:
            subgraph_nodes['features'].add(feat['feature_id'])

        # Hop 1: Expand feature-to-feature connections
        if n_hops >= 1:
            seed_features = list(subgraph_nodes['features'])

            for feat_id in seed_features:
                # Feature -> Feature edges (only from detected seed features)
                if feat_id in self.edge_index['feature_to_feature']:
                    for target_feat, weight, subtype in self.edge_index['feature_to_feature'][feat_id]:
                        if weight > edge_threshold:
                            # Add new feature node if not already present
                            if target_feat not in subgraph_nodes['features']:
                                subgraph_nodes['features'].add(target_feat)

                            subgraph_edges.append({
                                'source': feat_id,
                                'target': target_feat,
                                'edge_type': 'feature_to_feature',
                                'subtype': subtype,
                                'weight': weight
                            })

        # Hop 2: Expand from ALL features to semantic nodes, then to labels
        # Also add direct feature -> label edges
        if n_hops >= 2:
            # 2a: Feature -> Semantic
            for feat_id in list(subgraph_nodes['features']):
                if feat_id in self.edge_index['feature_to_semantic']:
                    for cat_id, weight in self.edge_index['feature_to_semantic'][feat_id]:
                        if weight > edge_threshold:
                            subgraph_nodes['semantics'].add(cat_id)
                            subgraph_edges.append({
                                'source': feat_id,
                                'target': cat_id,
                                'edge_type': 'feature_to_semantic',
                                'weight': weight
                            })

            # 2b: Semantic -> Label
            for cat_id in list(subgraph_nodes['semantics']):
                if cat_id in self.edge_index['semantic_to_label']:
                    for label_id, weight in self.edge_index['semantic_to_label'][cat_id]:
                        if weight > edge_threshold:
                            subgraph_nodes['labels'].add(label_id)
                            subgraph_edges.append({
                                'source': cat_id,
                                'target': label_id,
                                'edge_type': 'semantic_to_label',
                                'weight': weight
                            })

            # 2c: Feature -> Label
            for feat_id in list(subgraph_nodes['features']):
                if feat_id in self.edge_index['feature_to_label']:
                    for label_id, weight in self.edge_index['feature_to_label'][feat_id]:
                        if weight > edge_threshold:
                            subgraph_nodes['labels'].add(label_id)
                            subgraph_edges.append({
                                'source': feat_id,
                                'target': label_id,
                                'edge_type': 'feature_to_label',
                                'weight': weight
                            })

        # Prune if too many edges
        if len(subgraph_edges) > max_edges:
            # Sort by weight and keep top edges
            subgraph_edges.sort(key=lambda e: e['weight'], reverse=True)
            subgraph_edges = subgraph_edges[:max_edges]

            # Update nodes based on remaining edges
            remaining_nodes = {'features': set(), 'semantics': set(), 'labels': set()}
            for edge in subgraph_edges:
                source = edge['source']
                target = edge['target']

                if source.startswith('AF'):
                    remaining_nodes['features'].add(source)
                elif source.startswith('AC'):
                    remaining_nodes['semantics'].add(source)

                if target.startswith('AF'):
                    remaining_nodes['features'].add(target)
                elif target.startswith('AC'):
                    remaining_nodes['semantics'].add(target)
                elif target.startswith('CL'):
                    remaining_nodes['labels'].add(target)

            subgraph_nodes = remaining_nodes

        # Build node details
        node_details = {
            'features': [
                {
                    'feature_id': fid,
                    'feature_name': self.feature_nodes[fid]['feature_name'],
                    'detected': any(d['feature_id'] == fid for d in detected_features),
                    'confidence': next((d['confidence'] for d in detected_features if d['feature_id'] == fid), 0.0)
                }
                for fid in subgraph_nodes['features']
            ],
            'semantics': [
                {
                    'semantic_id': cid,
                    'semantic_name': self.semantic_nodes[cid]['semantic_name']
                }
                for cid in subgraph_nodes['semantics']
            ],
            'labels': [
                {
                    'label_id': lid,
                    'label_name': self.label_nodes[lid]['label_name']
                }
                for lid in subgraph_nodes['labels']
            ]
        }

        return {
            'nodes': node_details,
            'edges': subgraph_edges,
            'statistics': {
                'n_features': len(subgraph_nodes['features']),
                'n_semantics': len(subgraph_nodes['semantics']),
                'n_labels': len(subgraph_nodes['labels']),
                'n_edges': len(subgraph_edges),
                'detected_features': len([f for f in node_details['features'] if f['detected']])
            }
        }


# ================= DATA LOADING =================
def load_test_data():
    """Load test embeddings and dataframe"""
    test_embeddings = np.load(f'{DATA_DIR}/test_embeddings.npy')
    test_df = pd.read_csv(f'{DATA_DIR}/test_set.csv')

    # Ensure label column is integer
    test_df['label'] = test_df['label'].astype(int)

    return test_embeddings, test_df


def load_rag_pool():
    """Load RAG pool embeddings and dataframe"""
    rag_embeddings = np.load(f'{DATA_DIR}/rag_embeddings.npy')
    rag_df = pd.read_csv(f'{DATA_DIR}/rag_pool.csv')

    # Ensure label column is integer
    rag_df['label'] = rag_df['label'].astype(int)

    return rag_embeddings, rag_df


# ================= RAG RETRIEVAL =================
def retrieve_rag_examples(query_embedding, rag_embeddings, rag_df, top_k=TOP_K_TEXT):
    """Retrieve top-k most similar examples from RAG pool"""
    # Compute cosine similarities
    similarities = np.dot(rag_embeddings, query_embedding)

    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Get examples
    examples = []
    for idx in top_indices:
        examples.append({
            'text': rag_df.iloc[idx]['text'],
            'label': rag_df.iloc[idx]['label'],
            'label_name': ID_TO_LABEL[rag_df.iloc[idx]['label']],
            'similarity': similarities[idx]
        })

    return examples


# ================= LLM CLASSIFICATION =================
def classify_with_llm(text, rag_examples, evidence_subgraph, model_name, use_thinking=False):
    """
    Classify text using LLM with RAG examples and evidence subgraph

    Args:
        text: Input text to classify
        rag_examples: Retrieved RAG examples
        evidence_subgraph: Constructed evidence subgraph
        model_name: LLM model name
        use_thinking: Whether to use thinking mode (for Qwen models)

    Returns:
        Predicted label name and full response
    """

    # Build examples context
    examples_text = ""
    for i, sample in enumerate(rag_examples, 1):
        examples_text += f"\nExample {i} [{sample['label_name']}] (sim: {sample['similarity']:.3f}): {sample['text'][:200]}...\n"

    # Build subgraph context
    subgraph_text = ""
    if evidence_subgraph['statistics']['n_edges'] > 0:
        # Detected features
        seed_features = [f for f in evidence_subgraph['nodes']['features'] if f['detected']]
        if seed_features:
            subgraph_text += "\nDetected Attribution Features:\n"
            for feat in seed_features[:5]:
                subgraph_text += f"  - {feat['feature_name']} (confidence: {feat['confidence']:.2f})\n"

        # Connected semantic nodes
        if evidence_subgraph['nodes']['semantics']:
            subgraph_text += "\nRelated Attribution Categories:\n"
            for cat in evidence_subgraph['nodes']['semantics'][:3]:
                subgraph_text += f"  - {cat['semantic_name']}\n"

        # Linked labels
        if evidence_subgraph['nodes']['labels']:
            subgraph_text += "\nLinked Attribution Labels:\n"
            for label in evidence_subgraph['nodes']['labels']:
                subgraph_text += f"  - {label['label_name']}\n"

        # Key edges (top 5 by weight)
        subgraph_text += f"\nKey Evidence Edges (top 5):\n"
        sorted_edges = sorted(evidence_subgraph['edges'], key=lambda e: e['weight'], reverse=True)[:5]
        for edge in sorted_edges:
            # Get node names
            source_name = "Unknown"
            target_name = "Unknown"
            for f in evidence_subgraph['nodes']['features']:
                if f['feature_id'] == edge['source']:
                    source_name = f['feature_name']
                if f['feature_id'] == edge['target']:
                    target_name = f['feature_name']
            for c in evidence_subgraph['nodes']['semantics']:
                if c['semantic_id'] == edge['source']:
                    source_name = c['semantic_name']
                if c['semantic_id'] == edge['target']:
                    target_name = c['semantic_name']
            for l in evidence_subgraph['nodes']['labels']:
                if l['label_id'] == edge['source']:
                    source_name = l['label_name']
                if l['label_id'] == edge['target']:
                    target_name = l['label_name']
            subgraph_text += f"  - {source_name} → {target_name} (weight: {edge['weight']:.3f})\n"

    # Simple prompt with RAG and KG evidence
    prompt = f"""Classify this mental health attribution text using similar examples and knowledge graph:

Categories:
- no_reason
- bias_or_abuse
- jobs_and_careers
- medication
- relationship
- alienation

Similar Examples:
{examples_text}

Knowledge Graph Evidence:
{subgraph_text}

Text to classify: {text}

Respond with ONLY the category name."""

    try:
        options = {}
        if 'qwen' in model_name.lower() and use_thinking:
            options['enable_thinking'] = True

        client = ollama.Client(host=OLLAMA_HOST)
        response = client.generate(
            model=model_name,
            prompt=prompt,
            options=options
        )

        # Extract response
        if use_thinking and 'qwen' in model_name.lower():
            answer = response.get('response', '').strip()
            thinking = response.get('thinking', '')
            reasoning = f"Thinking: {thinking}\nAnswer: {answer}"
        else:
            answer = response.get('response', '').strip()
            reasoning = answer

        # Parse category
        answer_lower = answer.lower()
        predicted_label = None

        for label in ['no_reason', 'bias_or_abuse', 'jobs_and_careers', 'medication', 'relationship', 'alienation']:
            if label in answer_lower:
                predicted_label = label
                break

        if predicted_label is None:
            predicted_label = 'no_reason'

        return predicted_label, reasoning

    except Exception as e:
        print(f"Error in LLM classification: {e}")
        return 'no_reason', f"Error: {e}"


# ================= EVALUATION =================
def run_evaluation(test_embeddings, test_df, rag_embeddings, rag_df,
                   feature_detector, subgraph_builder, model_name, use_thinking=False):
    """
    Run complete evaluation pipeline

    Args:
        test_embeddings: Test set embeddings
        test_df: Test set dataframe
        rag_embeddings: RAG pool embeddings
        rag_df: RAG pool dataframe
        feature_detector: Feature detector instance
        subgraph_builder: Subgraph builder instance
        model_name: LLM model name
        use_thinking: Whether to use thinking mode

    Returns:
        Evaluation results dictionary
    """

    n_samples = len(test_df)

    y_true = []
    y_pred = []
    y_pred_labels = []
    interpretable_cases = []

    # Process each test sample
    for idx in tqdm(range(len(test_df)), desc="Classifying"):
        text = test_df.iloc[idx]['text']
        true_label = test_df.iloc[idx]['label']
        true_label_name = ID_TO_LABEL[true_label]

        query_embedding = test_embeddings[idx]

        # Step 1: Retrieve RAG examples
        rag_examples = retrieve_rag_examples(query_embedding, rag_embeddings, rag_df)

        # Step 2: Detect features and build subgraph
        detected_features = feature_detector.detect_features(text)
        evidence_subgraph = subgraph_builder.build_subgraph(detected_features)

        # Step 3: LLM classification
        predicted_label_name, llm_response = classify_with_llm(
            text, rag_examples, evidence_subgraph, model_name, use_thinking
        )

        predicted_label = LABEL_TO_ID[predicted_label_name]

        # Record results
        y_true.append(true_label)
        y_pred.append(predicted_label)
        y_pred_labels.append(predicted_label_name)

        # Save interpretable case
        interpretable_cases.append({
            'index': idx,
            'text': text,
            'true_label': int(true_label),
            'true_label_name': true_label_name,
            'predicted_label': int(predicted_label),
            'predicted_label_name': predicted_label_name,
            'is_correct': predicted_label == true_label,
            'rag_examples': rag_examples,
            'evidence_subgraph': evidence_subgraph,
            'llm_response': llm_response
        })

    # ================= COMPUTE METRICS =================
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # AUC (one-vs-rest)
    n_classes = len(ID_TO_LABEL)
    y_true_onehot = np.eye(n_classes)[y_true]
    y_pred_onehot = np.eye(n_classes)[y_pred]

    auc_per_class = []
    for i in range(n_classes):
        try:
            auc = roc_auc_score(y_true_onehot[:, i], y_pred_onehot[:, i])
        except:
            auc = 0.5  # Default for undefined cases
        auc_per_class.append(auc)

    auc_macro = np.mean(auc_per_class)

    # ================= PRINT RESULTS =================
    mode_label = "Thinking Mode" if use_thinking else "Standard Mode"

    print(f"\n{'=' * 70}")
    print(f"PER-CLASS METRICS ({mode_label})")
    print(f"{'=' * 70}")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12} {'Support':<10}")
    print(f"{'-' * 70}")

    for i, label_name in enumerate(['no_reason', 'bias_or_abuse', 'jobs_and_careers',
                                    'medication', 'relationship', 'alienation']):
        print(f"{label_name:<20} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {auc_per_class[i]:<12.4f} {support_per_class[i]:<10}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY METRICS ({mode_label})")
    print(f"{'=' * 70}")
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Macro Precision:   {precision_macro:.4f}")
    print(f"Macro Recall:      {recall_macro:.4f}")
    print(f"Macro F1-Score:    {f1_macro:.4f}")
    print(f"Macro AUC:         {auc_macro:.4f}")

    # ================= SAVE RESULTS =================

    # Separate correct and incorrect cases
    correct_cases = [c for c in interpretable_cases if c['is_correct']]
    incorrect_cases = [c for c in interpretable_cases if not c['is_correct']]

    # Filter cases with non-empty subgraphs (must have edges)
    def has_edges(case):
        """Check if the case has a non-empty subgraph with edges"""
        edges = case.get('evidence_subgraph', {}).get('edges', [])
        return len(edges) > 0

    correct_with_edges = [c for c in correct_cases if has_edges(c)]
    incorrect_with_edges = [c for c in incorrect_cases if has_edges(c)]

    # Select 10 correct and 10 incorrect cases with edges for saving
    selected_correct = correct_with_edges[:10] if len(correct_with_edges) >= 10 else correct_with_edges
    selected_incorrect = incorrect_with_edges[:10] if len(incorrect_with_edges) >= 10 else incorrect_with_edges



    results = {
        'experiment_config': {
            'model': model_name,
            'mode_label': mode_label,
            'use_thinking': use_thinking,
            'feature_detection_threshold': FEATURE_DETECTION_THRESHOLD,
            'edge_weight_threshold': EDGE_WEIGHT_THRESHOLD,
            'max_features_per_text': MAX_FEATURES_PER_TEXT,
            'n_hop_expansion': N_HOP_EXPANSION,
            'max_edges_per_subgraph': MAX_EDGES_PER_SUBGRAPH
        },
        'metrics': {
            'accuracy': float(acc),
            'macro_avg': {
                'precision': float(precision_macro),
                'recall': float(recall_macro),
                'f1': float(f1_macro),
                'auc': float(auc_macro)
            },
            'weighted_avg': {
                'precision': float(precision_weighted),
                'recall': float(recall_weighted),
                'f1': float(f1_weighted)
            },
            'per_class': {
                label_name: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1': float(f1_per_class[i]),
                    'auc': float(auc_per_class[i]),
                    'support': int(support_per_class[i])
                }
                for i, label_name in enumerate(['no_reason', 'bias_or_abuse', 'jobs_and_careers',
                                                'medication', 'relationship', 'alienation'])
            }
        },
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'case_counts': {
            'total': len(interpretable_cases),
            'total_correct': len(correct_cases),
            'total_incorrect': len(incorrect_cases),
            'correct_with_edges': len(correct_with_edges),
            'incorrect_with_edges': len(incorrect_with_edges),
            'saved_correct': len(selected_correct),
            'saved_incorrect': len(selected_incorrect)
        },
        'correct_cases': selected_correct,
        'incorrect_cases': selected_incorrect
    }

    # Save all results
    import os
    results_dir = '<path/to/results>'
    os.makedirs(results_dir, exist_ok=True)

    # Clean model name
    clean_model_name = re.sub(r'[^\w\-]', '_', model_name)
    clean_mode_label = re.sub(r'[^\w\-]', '_', mode_label.lower())

    output_filename = os.path.join(results_dir, f'subgraph_results_{clean_model_name}_{clean_mode_label}.json')
    abs_output_path = os.path.abspath(output_filename)

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"Results saved to: {abs_output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()

    return results


# ================= MAIN ENTRY POINT =================
if __name__ == "__main__":
    base_kg = load_base_kg()
    optimized_kg = load_optimized_kg()

    feature_detector = FeatureDetector(base_kg)
    subgraph_builder = EvidenceSubgraphBuilder(base_kg, optimized_kg)

    test_embeddings, test_df = load_test_data()
    rag_embeddings, rag_df = load_rag_pool()
    print(f"Test samples: {len(test_df)}, RAG pool: {len(rag_df)}")

    results = run_evaluation(
        test_embeddings, test_df, rag_embeddings, rag_df,
        feature_detector, subgraph_builder,
        MODEL_NAME, USE_THINKING_MODE
    )
