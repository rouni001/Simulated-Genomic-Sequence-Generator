import os
import torch
from torch.utils.data import Dataset
from Bio import SeqIO
from typing import List
from sequence_generation import KmerMappingHash

MAXKMERSCOUNT = 10000
SEQLENGTH = 100
KMERSIZE = 7


def extract_kmers(fasta_file: str, kmer_size: int) -> List[str]:
    """
    Extracts unique k-mers from sequences in the provided FASTA file.

    Parameters:
        fasta_file (str): Path to the FASTA file.
        kmer_size (int): Size of k-mers to extract.

    Returns:
        List[str]: A list of unique k-mers.
    """
    kmers_list = set()

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        if len(sequence) < kmer_size:
            continue

        for i in range(len(sequence) - kmer_size + 1):
            kmer = sequence[i:i + kmer_size]
            kmers_list.add(kmer)

    return list(kmers_list)


def read_fasta_and_generate_kmers(file_path: str, k: int = SEQLENGTH) -> List[str]:
    """
    Reads a FASTA file and extracts k-mers of specified length.

    Args:
        file_path (str): Path to the FASTA file.
        k (int): Length of the k-mer (default is SEQLENGTH).

    Returns:
        List[str]: A list of all k-mers of length k.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    kmers_list = []
    nucleotide_dict = {
        'A': 1, 'C': 1, 'G': 1, 'T': 1,
        'U': 0, 'N': 0, 'R': 0, 'Y': 0,
        'S': 0, 'W': 0, 'K': 0, 'M': 0,
        'B': 0, 'D': 0, 'H': 0, 'V': 0,
    }

    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq)
        for i in range(len(sequence) - k + 1):
            if len(kmers_list) >= MAXKMERSCOUNT:
                return kmers_list
            
            kmer = sequence[i:i + k]
            if all(nucleotide_dict.get(base, 0) == 1 for base in kmer):
                kmers_list.append(kmer)

    return kmers_list


class GenomicDataset(Dataset):
    def __init__(self, sequences, kmer_map_fn: KmerMappingHash, kmer_size: int =KMERSIZE):
        self.kmer_size = kmer_size
        self.kmer_map_fn = kmer_map_fn 
        self.data = [self.kmer_tokenize(seq) for seq in sequences]

    def kmer_tokenize(self, sequence):
        """
        Tokenizes a DNA sequence into k-mers (subsequences of length kmer_size).
        """
        kmers = [sequence[i:i+self.kmer_size] for i in range(0, len(sequence) - self.kmer_size + 1)]
        return [self.kmer_map_fn.get_idx(kmer) for kmer in kmers]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


