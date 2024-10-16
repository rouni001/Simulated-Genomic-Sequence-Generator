import torch
import torch.nn.functional as F
from typing import List


class KmerMappingHash:
    def __init__(self, kmer_list: List[str]):
        """
        Initializes the k-mer mapping hash.

        Parameters:
            kmer_list (List[str]): List of unique k-mers.
        """
        self.kmer_list = kmer_list
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmer_list)}

    def get_idx(self, kmer: str) -> int:
        """
        Gets the index of the k-mer.

        Parameters:
            kmer (str): The k-mer to find.

        Returns:
            int: Index of the k-mer, or -1 if not found.
        """
        return self.kmer_to_idx.get(kmer, -1)

    def get_kmer(self, idx: int) -> str:
        """
        Gets the k-mer corresponding to the index.

        Parameters:
            idx (int): The index of the k-mer.

        Returns:
            str: The k-mer at the specified index.
        """
        return self.kmer_list[idx]


def generate_sequences_autoregressively(model: torch.nn.Module, kmer_map_fn: KmerMappingHash, 
                                         nb_sequences: int, seed_sequence: str, 
                                         length_to_generate: int, kmer_size: int) -> List[str]:
    """
    Generates nucleotide sequences using the autoregressive approach.

    Args:
        model (torch.nn.Module): Trained Transformer model.
        kmer_map_fn (KmerMappingHash): Mapping from k-mers to indices.
        nb_sequences (int): Number of sequences to generate.
        seed_sequence (str): Seed sequence for generation.
        length_to_generate (int): Length of each generated sequence.
        kmer_size (int): Size of the k-mers used.

    Returns:
        List[str]: List of generated sequences.
    """
    model.eval()
    seed_kmers = [seed_sequence[i:i + kmer_size] for i in range(len(seed_sequence) - kmer_size + 1)]
    tokenized_kmers = [kmer_map_fn.get_idx(kmer) for kmer in seed_kmers]
    generated_kmers = tokenized_kmers.copy()
    current_sequence = torch.tensor(tokenized_kmers).unsqueeze(0)

    target_kmer_count = (nb_sequences * length_to_generate - kmer_size + 1)

    while len(generated_kmers) < target_kmer_count:
        if current_sequence.size(1) > SEQLENGTH:
            current_sequence = current_sequence[:, -SEQLENGTH:]

        logits = model(current_sequence)[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        last_kmer_token = current_sequence[0, -1].item()
        last_kmer = kmer_map_fn.get_kmer(last_kmer_token)
        last_kmer_bases = last_kmer[-(kmer_size - 1):]

        possible_kmers = [last_kmer_bases + base for base in 'ACGT']
        mapped_kmers = [kmer_map_fn.get_idx(kmer) for kmer in possible_kmers]
        possible_kmer_probs = [probs[0, mapped_kmer].item() for mapped_kmer in mapped_kmers]

        most_probable_kmer_idx = torch.argmax(torch.FloatTensor(possible_kmer_probs)).item()
        most_probable_kmer_token = mapped_kmers[most_probable_kmer_idx]
        current_sequence = torch.cat([current_sequence, torch.tensor([[most_probable_kmer_token]], dtype=torch.long)], dim=1)
        generated_kmers.append(most_probable_kmer_token)

    return [kmer_map_fn.get_kmer(token.item()) for token in torch.tensor(generated_kmers)]


