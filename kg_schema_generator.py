"""
Psychology-Guided Knowledge Graph Construction for Psy-GRAG
============================================================
This script illustrates the two-stage LLM-assisted pipeline used to construct
the task-specific heterogeneous knowledge graph described in Section 3.1-3.2
of the paper. It is intended as a methodological reference, not a standalone
runnable script.

The Stage I and Stage II prompts below are representative examples showing
the structural constraints and psychological grounding used during construction.

Note: API credentials are not included. Set the DEEPSEEK_API_KEY environment
variable if running this script.
"""

import json
import os
from typing import Dict, Any, Optional


# =============================================================================
# LLM API CALL (abstracted)
# =============================================================================

def call_llm(prompt: str, model: str = "deepseek-reasoner") -> str:
    """
    Call the LLM API with the given prompt.
    Requires DEEPSEEK_API_KEY to be set as an environment variable.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in clinical psychology, mental health attribution theory, "
                    "and computational psychiatry. Output ONLY valid JSON. "
                    "Do NOT include any 'weight' fields in nodes or edges."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=64000
    )
    return response.choices[0].message.content


def parse_json_response(response: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    # Remove ```json ... ``` wrappers that LLMs sometimes add
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


# =============================================================================
# STAGE I PROMPT — Skeletal Graph Generation
# Constructs: all nodes (Layers 1-3) and inter-layer edges (f->s, s->l)
# Psychological grounding: Joiner’s Interpersonal Theory, DSM-IV, ICD-11, Beck’s Cognitive Triad, Attribution Theory, Hierarchical Psychiatric Taxonomy

# =============================================================================

STAGE_I_PROMPT = """
Role: You are a distinguished Professor of Clinical Psychology and Mental Health
Attribution Theory, specializing in computational psychiatry.

Task: Construct a 3-Layer Heterogeneous Knowledge Graph for mental health text
classification. The graph models the cognitive reasoning path from surface
linguistic expressions to diagnostic labels.

## Graph Architecture

Layer 1 — Linguistic Feature Layer:
  - 68-78 fine-grained attribution feature nodes
  - Each node: feature_id, feature_name, definition, informal_expressions (10-15),
    valence, severity, feature_type, observable, primary_semantic
  - Grounded in Joiner's Interpersonal Theory and DSM-IV diagnostic criteria

Layer 2 — Intermediate Semantic Layer:
  - 6 mid-level attribution semantic nodes
  - Each node: semantic_id, semantic_name, definition, primary_label,
    secondary_labels, mediates
  - Grounded in Beck's Cognitive Triad and Attribution Theory

Layer 3 — Task Label Layer:
  - One node per classification label
  - Each node: label_id, label_name, label_code, definition, characteristics,
    clinical_significance
  - Grounded in Hierarchical Psychiatric Taxonomy

## Edge Schema (Stage I)

feature_to_semantic edges (V_f -> V_s):
  Each feature connects to its primary category via "belongs_to" relation.
  Grounded in DSM-IV and ICD-11 diagnostic criteria.

semantic_to_label edges (V_s -> V_l):
  Each category connects to its primary and secondary labels via
  "primary_indicator" or "secondary_indicator" relations.

## Critical Constraints

1. Node Taxonomy Alignment: all node IDs must follow predefined schemas
   (e.g., AF001-AF078 for features, AC001-AC006 for categories, CL001-CL00N for labels)
2. Linguistic Enrichment: each feature node must include 10-15 informal expressions
   reflecting natural social media language (e.g., Reddit posts)
3. Axiomatic Schema Design: inter-layer edges must follow theory-grounded
   category assignments from DSM-IV and ICD-11
4. Weight-free Initialization: all edges are binary priors {{0, 1}};
   NO weight fields — confidence scores are learned during training

## Target Dataset

[Insert dataset-specific label definitions and classification schema here.
 Provide label names, codes, definitions, and clinical significance.]

## Output Format

Return ONLY valid JSON:
{{
  "metadata": {{ "graph_name": "...", "version": "1.0", "edge_weights": "NONE" }},
  "label_nodes": [...],
  "semantic_nodes": [...],
  "feature_nodes": [...],
  "edges": [...]
}}
"""


# =============================================================================
# STAGE II PROMPT — Relational Augmentation
# Adds: intra-layer f2f edges and shortcut f2l edges
# Psychological grounding: Stress-Diathesis Model, Cognitive Mediation Theory
# =============================================================================

STAGE_II_PROMPT = """
You are augmenting an existing knowledge graph with two additional edge types.
Do NOT include any "weight" fields. Weights will be learned during training.

## Existing Graph Summary
- {num_features} feature nodes (Layer 1), IDs: {feature_ids_sample}
- {num_semantics} semantic nodes (Layer 2)
- {num_labels} label nodes (Layer 3)
- Existing edges: feature_to_semantic, semantic_to_label

## Task: Generate Two New Edge Types

### Part 1 — Feature-to-Feature Intra-layer Edges (V_f <-> V_f)
Grounded in the Stress-Diathesis Model. Generate 60-80 edges across 5 subtypes:

1. co-occurs-in (25-30, undirected): features frequently co-occurring in mental health text
2. precedes (12-18, directed): temporal triggering sequences between features
3. intensifies (12-18, directed): feature A amplifies the severity of feature B
4. contradicts (5-8, undirected): mutually exclusive causal patterns
5. overlaps-with (10-15, undirected): conceptually similar but clinically distinguishable

Edge format:
{{
  "edge_id": "E_F2F_001",
  "edge_type": "feature_to_feature",
  "subtype": "co-occurs-in",
  "source_id": "AF001", "source_name": "...",
  "target_id": "AF008", "target_name": "...",
  "direction": "undirected",
  "rationale": "..."
}}

### Part 2 — Shortcut Feature-to-Label Edges (V_f -> V_l)
Grounded in Cognitive Mediation Theory. Add direct edges ONLY for features that
are strong (pathognomonic) diagnostic indicators of a specific label.
Generate 25-35 edges.

Edge format:
{{
  "edge_id": "E_F2L_001",
  "edge_type": "feature_to_label",
  "source_id": "AF003", "source_name": "...",
  "target_id": "CL002", "target_name": "...",
  "relationship_type": "strong_diagnostic",
  "direction": "directed",
  "theoretical_basis": "..."
}}

## Constraints
- Inventory-based Consistency: use only node IDs from the existing graph
- Distributional Balancing: all 5 f2f subtypes must be represented
- Pathognomonic Shortcut Rules: f2l edges restricted to highly diagnostic features
- NO weight fields

## Existing Nodes
Features:
{feature_list}

Labels:
{label_list}

Return ONLY valid JSON: {{ "new_edges": [...] }}
"""


# =============================================================================
# PIPELINE
# =============================================================================

def build_base_graph(model: str = "deepseek-reasoner") -> Dict[str, Any]:
    """
    Stage I: Generate all nodes and inter-layer edges via LLM.
    """
    print("Stage I: Skeletal graph generation...")
    response = call_llm(STAGE_I_PROMPT, model)
    kg = parse_json_response(response)

    # Ensure no weight fields slipped through
    for edge in kg.get("edges", []):
        edge.pop("weight", None)

    print(f"  Labels: {len(kg.get('label_nodes', []))}")
    print(f"  Semantics: {len(kg.get('semantic_nodes', []))}")
    print(f"  Features: {len(kg.get('feature_nodes', []))}")
    print(f"  Edges: {len(kg.get('edges', []))} (no weights)")
    return kg


def augment_graph(kg: Dict[str, Any], model: str = "deepseek-reasoner") -> Dict[str, Any]:
    """
    Stage II: Add intra-layer f2f edges and shortcut f2l edges.
    The full node inventory from Stage I is passed as context to prevent entity hallucination .
    """
    print("Stage II: Relational augmentation...")

    features = kg["feature_nodes"]
    labels = kg["label_nodes"]

    # Pass full node inventory to the LLM for consistency
    feature_list = "\n".join(
        f"- {f['feature_id']}: {f['feature_name']} (category: {f.get('primary_semantic', '?')})"
        for f in features
    )
    label_list = "\n".join(
        f"- {l['label_id']}: {l['label_name']}"
        for l in labels
    )
    feature_ids_sample = ", ".join(f["feature_id"] for f in features[:5]) + ", ..."

    prompt = STAGE_II_PROMPT.format(
        num_features=len(features),
        feature_ids_sample=feature_ids_sample,
        num_semantics=len(kg["semantic_nodes"]),
        num_labels=len(labels),
        feature_list=feature_list,
        label_list=label_list
    )

    response = call_llm(prompt, model)
    new_edges_data = parse_json_response(response)

    for edge in new_edges_data.get("new_edges", []):
        edge.pop("weight", None)

    kg["edges"].extend(new_edges_data["new_edges"])
    print(f"  Added {len(new_edges_data['new_edges'])} edges")
    print(f"  Total edges: {len(kg['edges'])} (no weights)")
    return kg


def build_kg(output_file: str, model: str = "deepseek-reasoner") -> Dict[str, Any]:
    """
    Full two-stage pipeline: skeletal generation followed by relational augmentation.
    The resulting graph is saved to output_file and used as input for kg_encode.py.
    """
    # Stage I: nodes + inter-layer edges
    kg = build_base_graph(model)

    # Stage II: intra-layer f2f edges + shortcut f2l edges
    kg = augment_graph(kg, model)

    # Final validation: remove any residual weight fields
    for edge in kg["edges"]:
        edge.pop("weight", None)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

    print(f"\nKnowledge graph saved to: {output_file}")
    return kg


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # API key must be set as an environment variable
    # e.g. export DEEPSEEK_API_KEY="your-key-here"
    build_kg(output_file="<path/to/output_kg.json>", model="deepseek-reasoner")