# GPCR-GO

<p align="center">
  <img src="assets/model-overview.png" alt="Overview of the GPCR-GO framework" width="100%" />
</p>

GPCR-GO is a relation-aware heterogeneous graph learning framework for predicting Gene Ontology (GO) terms of G protein-coupled receptors (GPCRs). This repository contains the training and evaluation code corresponding to our paper **"GPCR-GO: Relation-Aware Graph Learning for Predicting Gene Ontology Terms of G Protein-Coupled Receptors"**.

The model integrates sequence representations, structure-derived similarity, protein-protein interactions (PPIs), GO hierarchy relations, and protein-GO annotations into a unified heterogeneous graph. On top of the graph encoder, GPCR-GO further introduces graph decomposition regularization, semi-supervised learning with unlabeled GPCRs, and hard negative mining to improve prediction under sparse supervision and strong label imbalance.

This repository releases the preprocessed graph datasets used in the manuscript. The raw databases, filtering criteria, and feature-construction procedures for UniProt, GO, BioGRID, PDB, and AlphaFold are described in the Materials and methods section of the paper. The commands below reproduce the training and evaluation workflow from the released preprocessed graph datasets to the main reported metrics.

## 1. Highlights

- GPCR-specific heterogeneous graph containing reviewed proteins, unreviewed proteins, and GO term nodes.
- Multi-source biological evidence fusion, including ESM2-based sequence features, DSSP-derived structure similarity, PPIs, and GO graph relations.
- Relation-aware graph attention with edge-type embeddings and relation-specific weighting.
- Hard negative mining and weighted BCE optimization for severe positive/negative imbalance.
- Graph decomposition regularization to encourage complementary factor subspaces.
- CAFA-style evaluation with `Fmax`, `Smin`, and `AUPR`, plus additional metrics implemented in code.

## 2. Environment

The codebase is based on PyTorch and DGL. The paper experiments were run with the following software stack:

- Python 3.8
- PyTorch 1.12.1
- DGL 0.9.1.post1
- NumPy 1.23.5
- SciPy 1.9.3
- NetworkX 2.8.4
- scikit-learn 1.2.0

A minimal CPU environment can be prepared as follows:

```bash
conda create -n gpcr-go python=3.8
conda activate gpcr-go
pip install torch==1.12.1 numpy==1.23.5 scipy==1.9.3 networkx==2.8.4 scikit-learn==1.2.0
pip install dgl==0.9.1.post1
```

The training code supports both CPU and CUDA devices, although CUDA is strongly recommended for full experiments. If you use a CUDA-enabled environment, install the PyTorch and DGL builds matching your CUDA version.

## 3. Released Data

The repository provides the preprocessed BP, CC, and MF graph datasets used for the manuscript experiments in `data/reviewed6.rar`. Extract the archive into `data/` before training. For example, use either 7-Zip or `unrar` from the repository root:

```bash
7z x data/reviewed6.rar -odata
# or
unrar x data/reviewed6.rar data/
```

After extraction, the directory layout is:

```text
data/reviewed6/
|- bp/
|- cc/
`- mf/
```

The archive is already preprocessed; only extraction is required. Each task directory contains:

```text
dataset_metadata.json
node.dat
link.dat
link.dat.val
link.dat.test
protein2id.txt
id2protein.txt
go2id.txt
id2go.txt
protein_splits.tsv
terms.txt
ic_dict.json
```

According to `scripts/data_loader.py`, the graph files use the following text formats:

- `node.dat`: `node_id<TAB>node_name<TAB>node_type<TAB>feature`
- `link.dat`: `head_id<TAB>tail_id<TAB>relation_type<TAB>weight`
- `link.dat.val`: `head_id<TAB>tail_id<TAB>relation_type<TAB>weight`
- `link.dat.test`: `head_id<TAB>tail_id<TAB>relation_type<TAB>weight`

Notes:

- If a node line has only three columns, the loader will treat that node type as featureless and automatically use an identity matrix.
- In the released GPCR datasets, the first node type is used as proteins and the second node type is used as GO terms, which matches the indexing logic in `methods/model/runepochgpu1_v1_log.py`.
- The `reviewed6` package is already preprocessed. You do not need to run ESM2 or DSSP during training if you directly use the released graph files.
- Unreviewed protein nodes are always included in message passing and in the all-node graph decomposition regularization. There is no command-line option to disable them.

## 4. Data Splits

The released graph package contains the fixed 8:1:1 split used by the training and evaluation code. With random seed 42, the 3,465 reviewed proteins are divided into 2,772 training proteins, 346 validation proteins, and 347 test proteins.

- `protein_splits.tsv` records the train, validation, and test assignment of each reviewed protein.
- `link.dat` contains training protein-GO annotations and the auxiliary graph relations used for message passing.
- `link.dat.val` contains the fixed validation protein-GO annotations used for validation loss and model selection.
- `link.dat.test` contains the held-out test protein-GO annotation edges used for final evaluation.
- `dataset_metadata.json` records branch-specific protein, annotation, relation-edge, and GO-term counts.
- During final evaluation, `methods/model/runepochgpu1_v1_log.py` loads the best checkpoint selected by validation loss and evaluates it on `link.dat.test`.

The loader uses `link.dat.val` directly when it is present, so the released training, validation, and test annotations remain fixed across runs.

## 5. Quick Start

After extracting the archive, the released data are located under `data/reviewed6/`. Run the commands from `methods/model/`:

```bash
cd methods/model
```

Run BP:

```bash
python runepochgpu1_v1_log.py --dataset reviewed6/bp
```

Run CC:

```bash
python runepochgpu1_v1_log.py --dataset reviewed6/cc
```

Run MF:

```bash
python runepochgpu1_v1_log.py --dataset reviewed6/mf
```

The script selects the manuscript hyperparameters from the final `bp`, `cc`, or `mf` component of `--dataset`: BP uses `Pe=15`, `Pn=15`, `M=8`, and `lambda=0.1`; CC uses `Pe=10`, `Pn=15`, `M=2`, and `lambda=0.1`; MF uses `Pe=15`, `Pn=15`, `M=4`, and `lambda=1.0`. Hard negative mining is enabled by default. These values can still be overridden with command-line options.

The main shared defaults are learning rate `5e-4`, weight decay `2e-4`, batch size `64`, dropout `0.5`, maximum epochs `1000`, early-stopping patience `100`, hidden dimension `64`, two graph-attention layers followed by an attention-based output projection, four attention heads per stage, hard-negative fraction `0.5`, hard-negative candidate pool size `512`, and random seed `42`.

## 6. Outputs and Main Metrics

During training, the script writes checkpoints under `methods/model/checkpoint/paper2/<run_id>/` and appends run metadata to `methods/model/training_logs.jsonl`. By default, only the best checkpoint selected by validation loss is retained and evaluated on `link.dat.test`.

The terminal output includes CAFA-style metrics, including:

- `fmax`: maximum protein-centric F1 score across thresholds.
- `smin`: minimum semantic distance.
- `aupr`: area under the precision-recall curve.

These are the metrics reported in the main performance table of the manuscript for BP, MF, and CC.

Because GPU kernels and library versions may introduce small numerical variation, reproduced values can differ slightly across hardware and software builds. The released data, split logic, hyperparameters, and evaluation code are provided to reproduce the reported workflow from the preprocessed graph datasets.
