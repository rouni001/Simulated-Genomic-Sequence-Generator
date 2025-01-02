# Simulated Genomic Sequence Generator

This package generates simulated genomic sequences using a Transformer-based architecture trained on reference genomes.

## Author

- **Rachid Ounit, Ph.D.** 

## Date

- **October 15, 2024**

## Introduction

In genomic research, access to large, diverse, and high-quality datasets is crucial for developing robust algorithms and models. However, obtaining real genomic data can be challenging due to issues like privacy concerns, data scarcity, or ethical restrictions. Synthetic genomic datasets play a critical role in overcoming these limitations. By generating artificial data that closely resembles real genomic sequences, researchers can test and validate models in a controlled environment, refine bioinformatics tools, and simulate experimental conditions. The ability to generate synthetic datasets is particularly useful for applications such as testing sequence alignment algorithms, validating variant calling methods, and improving machine learning models for genomic prediction tasks.

While many simulators of genomic sequences based on traditional approaches exist, we present here a **Simulated Genomic Sequence Generator** as a Python-based framework designed to generate synthetic genomic sequences using self-supervised learning techniques, specifically leveraging Transformer architectures. It aimed at bioinformatics applications, such as testing, validation, and the generation of synthetic genomic data for experimental purposes. The system uses real genomic sequences to train a Transformer model, which then generates new, synthetic sequences that preserve the statistical properties and structure of the original data.

This tool is also provided as a prototype in order to assess how feasible/realistic it is to generate synthetic genomic sequences through a Transformer model.

## Features

- **Synthetic Genomic Sequence Generation**: Autoregressively generates genomic sequences, one k-mer at a time.
- **Transformer-based Model**: Utilizes state-of-the-art Transformer architecture to capture long-range dependencies in genomic data.
- **K-mer Extraction**: Automatically extracts k-mers (subsequences of fixed length k) from reference genomes to prepare training data.
- **Flexible Sequence Generation**: Generates synthetic sequences of variable lengths, mimicking the structure of the training data.

## Dependencies

This package requires python3.9 and the following packages:
- `torch`
- `torchvision`
- `biopython`
- `numpy`
- `argparse`

## Installation

To install the necessary dependencies, run the following command:
```bash
pip3 install -r requirements.txt
```

## Modules

### data_processing.py

This module contains functions for extracting k-mers from FASTA files.

#### Functions:
- **extract_kmers(fasta_file: str, kmer_size: int) -> List[str]**: 
  Extracts unique k-mers from sequences in the provided FASTA file.

- **read_fasta_and_generate_kmers(file_path: str, k: int = SEQLENGTH) -> List[str]**: 
  Reads a FASTA file and extracts k-mers of specified length.

---

### model.py

This module defines the Transformer model for sequence generation.

#### Classes:
- **TransformerSequenceModel**: 
  Initializes the Transformer sequence model with specified parameters.

  ##### Methods:
  - **__init__(self, input_dim: int, embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, sequence_length: int)**: 
    Initializes the model's architecture.

  - **_initialize_weights(self)**: 
    Initializes weights using Xavier uniform distribution.

  - **get_sinusoidal_positional_encoding(sequence_length: int, embed_dim: int) -> torch.Tensor**: 
    Generates sinusoidal positional encodings.

  - **forward(self, x: torch.Tensor) -> torch.Tensor**: 
    Forward pass through the model.

---

### training.py

This module contains functions to train the model on genomic data.

#### Functions:
- **train_model(model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, criterion: nn.Module, epochs: int, device: str, scheduler: Optional[Optimizer] = None) -> None**: 
  Trains the model on the provided dataset.

---

### sequence_generation.py

This module implements autoregressive sequence generation using the trained model.

#### Classes:
- **KmerMappingHash**: 
  Initializes the k-mer mapping hash.

  ##### Methods:
  - **__init__(self, kmer_list: List[str])**: 
    Initializes k-mer mapping from a list.

  - **get_idx(self, kmer: str) -> int**: 
    Gets the index of the k-mer.

  - **get_kmer(self, idx: int) -> str**: 
    Gets the k-mer corresponding to the index.

#### Functions:
- **generate_sequences_autoregressively(model: nn.Module, kmer_map_fn: KmerMappingHash, nb_sequences: int, seed_sequence: str, length_to_generate: int, kmer_size: int) -> List[str]**: 
  Generates nucleotide sequences using the autoregressive approach.

---

## Usage

### K-mer Extraction

The first step in using the system is extracting k-mers from your genomic data (FASTA file format). This is done using the provided script `data_processing.py`.

```bash
python data_processing.py --input <input_genomic_file> --output <output_kmers_file> --kmer_length <kmer_size>
```

### Model Training

Once the k-mers are extracted, you can train the model using the `training.py` script. This will train a Transformer model to generate sequences based on the extracted k-mers.

```bash
python training.py --kmer_file <output_kmers_file> --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate>
```

### Sequence Generation

After training the model, you can generate synthetic genomic sequences using the `sequence_generation.py` script. The model will predict new k-mers iteratively, extending the sequence one k-mer at a time.

```bash
python sequence_generation.py --model_path <trained_model_path> --length <sequence_length> --output <generated_sequences_file>
```

## Autoregressive Sequence Generation

The core of this project lies in its autoregressive sequence generation mechanism:

- **Initialization**: The generation process begins by extracting an initial set of k-mers from a reference genomic sequence.
- **Autoregressive Process**: The Transformer model predicts the next k-mer based on the previous k-mers. This is done by considering the first `k-1` bases of the last k-mer and selecting the next k-mer with the highest softmax probability.
- **Iterative Extension**: The predicted k-mer is appended to the existing sequence, and the process repeats until the desired sequence length is achieved.
- **Effectiveness**: This autoregressive method is highly effective for generating synthetic sequences that preserve key genomic properties, such as GC content and k-mer distributions. It captures local dependencies between k-mers, making the generated sequences realistic and useful for various applications.

## Execution

To generate the simulated genomic sequences (using default parameters), run the main script:
```bash
python3 main.py
```

## License

This project is licensed under the MIT License.


