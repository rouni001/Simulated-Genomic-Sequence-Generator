# Genomic Sequence Generator

This package generates simulated genomic sequences using a Transformer-based architecture trained on reference genomes.

## Goal

The primary goal of this repository is to provide a framework for generating in-silico genomic sequences based on k-mer extraction and a Transformer model. This can be useful for various bioinformatics applications, such as simulating genomic data for testing and validation purposes.

## Author

- **Rachid Ounit, Ph.D.** 

## Date

- **October 15, 2024**

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

To generate the simulated genomic sequences (using default parameters), run the main script:
```bash
python3 main.py
```

## License

This project is licensed under the MIT License.


