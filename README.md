# GPCR-GO

<p align="center">
  <img src="assets/model-overview.png" alt="Overview of the GPCR-GO framework" width="100%" />
</p>

GPCR-GO is a relation-aware heterogeneous graph learning framework for predicting Gene Ontology (GO) terms of G protein-coupled receptors (GPCRs). This repository contains the training and evaluation code corresponding to our paper **"GPCR-GO: Relation-Aware Graph Learning for Predicting Gene Ontology Terms of G Protein-Coupled Receptors"**.

The model integrates sequence representations, structure-derived similarity, protein-protein interactions (PPIs), GO hierarchy relations, and protein-GO annotations into a unified heterogeneous graph. On top of the graph encoder, GPCR-GO further introduces graph decomposition regularization, semi-supervised learning with unlabeled GPCRs, and hard negative mining to improve prediction under sparse supervision and strong label imbalance.

> Note: this repository focuses on model training/evaluation on the released graph datasets. The raw preprocessing workflow from UniProt/PDB/AlphaFold/STRING to the final graph files is described in the paper, while the released `reviewed5` package already contains the processed inputs expected by the code.

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

A minimal environment can be prepared as follows:

```bash
conda create -n gpcr-go python=3.8
conda activate gpcr-go
pip install torch==1.12.1 numpy==1.23.5 scipy==1.9.3 networkx==2.8.4 scikit-learn==1.2.0
pip install dgl==0.9.1.post1
```

If you use a CUDA-enabled environment, please install the PyTorch and DGL builds matching your CUDA version.

## 3. Data

According to `scripts/data_loader.py`, the released graph package uses the following text formats:

- `node.dat`: `node_id<TAB>node_name<TAB>node_type<TAB>feature`
- `link.dat`: `head_id<TAB>tail_id<TAB>relation_type<TAB>weight`
- `link.dat.test`: `head_id<TAB>tail_id<TAB>relation_type<TAB>weight`

Notes:

- If a node line has only three columns, the loader will treat that node type as featureless and automatically use an identity matrix.
- In the released GPCR datasets, the first node type is used as proteins and the second node type is used as GO terms, which matches the indexing logic in `methods/model/run.py`.
- The `reviewed5` package is already preprocessed. You do not need to run ESM2 or DSSP during training if you directly use the released graph files.

## 4. Quick Start

Please run the commands from `methods/model/` so that the relative data path resolves correctly:

```bash
cd methods/model
python run.py --dataset reviewed5/bp --hardneg
python run.py --dataset reviewed5/mf --hardneg
python run.py --dataset reviewed5/cc --hardneg
```

These commands train and evaluate BP, MF, and CC separately.
