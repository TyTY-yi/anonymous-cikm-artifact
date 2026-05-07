
This repository provides the anonymous implementation and reproducibility materials for **Psy-GRAG**.

---

## Repository Structure

    .
    ├── data_process.py              # Data loading, deduplication, splitting, and MentalBERT encoding
    ├── kg_schema_generator.py        # Two-stage LLM-assisted KG construction protocol
    ├── kg_encode.py                  # Node encoding and adjacency matrix construction
    ├── kg_optimizer.py               # Multi-round Gumbel-Sigmoid edge weight optimization
    ├── subgraph_reasoning_main.py    # Evidence subgraph retrieval and LLM-assisted classification
    ├── cams_kg.json                 # CAMS knowledge graph used in the experiments
    └── README.md

---

## Requirements

We recommend using Python 3.9 or later.

Install the required Python packages:

    pip install torch transformers sentence-transformers scikit-learn pandas numpy openai tqdm ollama

LLM inference is run locally through Ollama. Install Ollama and pull the required model before running inference:

    ollama pull qwen2.5:7b, llama3:8b, mistral:7b, gemma:7b

The embedding model used in preprocessing and KG encoding is:

    mental/mental-bert-base-uncased

---

## Datasets

We evaluate Psy-GRAG on publicly available mental health text classification datasets.

### CAMS

Reddit mental health attribution dataset with 6 attribution labels.

    https://github.com/drmuskangarg/CAMS

### SWMH

Multi-subreddit mental health classification dataset.

    https://huggingface.co/datasets/AIMH/SWMH

### MultiWD

Multi-label wellness dimension classification dataset.

    https://github.com/drmuskangarg/MultiWD

---

## Pipeline

The full pipeline consists of five stages.

---

## Step 1: Data Preprocessing

This step merges the original dataset splits, removes duplicated texts, constructs a new stratified train/test/RAG split, and encodes all texts using MentalBERT.

The resulting split is:

    50% training set
    25% test set
    25% RAG pool

The RAG pool is separated from the test set and is used only for retrieval augmentation during inference.

Set the following paths in `data_process.py`:

    DATA_DIR = "<path/to/dataset>"
    OUTPUT_DIR = "<path/to/processed_data>"

Run:

    python data_process.py

Expected outputs:

    train_set.csv
    test_set.csv
    rag_pool.csv
    train_embeddings.npy
    test_embeddings.npy
    rag_embeddings.npy

---

## Step 2: Knowledge Graph Construction

The script `kg_schema_generator.py` provides a reference implementation of the two-stage LLM-assisted KG construction protocol. Because the graph schema and label definitions are dataset-specific, the prompts and configuration should be adapted to the target dataset before constructing a new KG. 

For the CAMS experiments, we provide the anonymized constructed KG directly at:

cams_kg.json

The construction process consists of:

1. **Stage I:** Generate label nodes, semantic nodes, feature nodes, and inter-layer edges.
2. **Stage II:** Add feature-to-feature relational edges and shortcut feature-to-label edges.

If running the script with a compatible OpenAI-style API endpoint, set the required API key first:

    export DEEPSEEK_API_KEY="your-key-here"

Then configure the output path inside `kg_schema_generator.py` and run:

    python kg_schema_generator.py

Expected output:

    kg.json

---

## Step 3: Knowledge Graph Encoding

This step encodes all KG nodes using MentalBERT and builds adjacency matrices for all heterogeneous edge types.

Set the following paths in `kg_encode.py`:

    INPUT_KG_PATH = "kg/cams_kg.json"
    OUTPUT_DIR = "<path/to/processed_data>"

Run:

    python kg_encode.py

Expected outputs:

    feature_embeddings.npy
    semantic_embeddings.npy
    label_embeddings.npy
    adj_f2s.npy
    adj_s2l.npy
    adj_f2l.npy
    adj_f2f_co_occurs.npy
    adj_f2f_precedes.npy
    adj_f2f_intensifies.npy
    adj_f2f_contradicts.npy
    adj_f2f_overlaps.npy
    kg_metadata.json
    node_index_mappings.json

---

## Step 4: Graph Optimization

This step learns task-adaptive edge weights through multi-round Gumbel-Sigmoid optimization.

The optimizer runs multiple independent rounds with different random seeds, averages the learned edge weights, and prunes low-confidence edges to obtain the optimized sparse graph.

Set the following paths in `kg_optimizer.py`:

    DATA_DIR = "<path/to/processed_data>"
    KG_DIR = "<path/to/processed_data>"
    OUTPUT_DIR = "<path/to/optimized_kg_results>"
    VISUALIZATION_DIR = "<path/to/visualizations>"

Run:

    python kg_optimizer.py

Expected outputs:

    adj_*_optimized.npy
    optimized_kg_weighted.json
    training_histories.json

---

## Step 5: Inference and Evaluation

This step constructs input-adaptive evidence subgraphs, retrieves semantically similar examples from the RAG pool, and performs LLM-assisted classification.

Set the following paths in `subgraph_reasoning_main.py`:

    DATA_DIR = "<path/to/processed_data>"
    OPTIMIZED_KG_DIR = "<path/to/optimized_kg_results>"
    BASE_KG_PATH = "kg/cams_kg.json"
    MODEL_NAME = "qwen2.5:7b"

Make sure Ollama is running locally:

    ollama serve

Then run:

    python subgraph_reasoning_main.py

Expected outputs include:

    classification results
    per-class metrics
    summary metrics
    accuracy
    macro-F1
    saved evidence subgraphs

---

## Anonymity
This repository is prepared for anonymous peer review. It does not include author names, affiliations, acknowledgments, institutional paths, or non-anonymized project metadata.




