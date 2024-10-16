import torch
import os
import argparse
from data_processing import (
    KMERSIZE, 
    GenomicDataset, 
    read_fasta_and_generate_kmers, 
    extract_kmers
)
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerSequenceModel
from training import train_model
from sequence_generation import KmerMappingHash, generate_sequences_autoregressively


def main(embed_dim: int, num_heads: int, hidden_dim: int, num_layers: int, 
         lr: float, epochs: int, batch_size: int, seed_sequence: str, 
         length_to_generate: int, nb_sequences: int, ref_sequences: str) -> None:
    """
    Main function to orchestrate the training and sequence generation process.

    Parameters:
        embed_dim (int): Dimensionality of the embedding layer.
        num_heads (int): Number of attention heads in the Transformer model.
        hidden_dim (int): Dimensionality of the feedforward layer.
        num_layers (int): Number of layers in the Transformer model.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        seed_sequence (str): Seed sequence for generation.
        length_to_generate (int): Length of each generated sequence.
        nb_sequences (int): Number of sequences to generate.
        reference_sequences (str): File path to the training sequences.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sequences = read_fasta_and_generate_kmers(ref_sequences)
    print(f"# kmers for training: {len(sequences)}")
    seq_length = len(sequences[0])

    kmers_list = extract_kmers(ref_sequences, KMERSIZE)
    kmer_mapping = KmerMappingHash(kmers_list)

    dataset = GenomicDataset(sequences, kmer_mapping)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 37007
    model = TransformerSequenceModel(input_dim, embed_dim, num_heads, hidden_dim, num_layers, seq_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()

    checkpoint_path = f"model_d{embed_dim}.b{batch_size}.e{epochs}.k{KMERSIZE}.chkpt.pth"
    if os.path.exists(checkpoint_path):
        print("Loading pre-trained model from disk...")
        model, optimizer, _, _, _ = load_model(checkpoint_path, model, optimizer)
    else:
        print("Starting training from scratch...")
        train_model(model, dataloader, optimizer, criterion, epochs, device)
        save_model(model, optimizer, epochs, 0, checkpoint_path)

    print("Generating sequences...")
    generated_sequence = generate_sequences_autoregressively(
        model, kmer_mapping, nb_sequences, seed_sequence, length_to_generate, KMERSIZE
    )
    
    for i, seq in enumerate(generated_sequence):
        print(f">Seq{i + 1}\n{seq}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genomic Sequence Generator")
    parser.add_argument("--embed_dim", type=int, default=48, help="Dimensionality of the embedding layer.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimensionality of the feedforward layer.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers in the Transformer.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--seed_sequence", type=str, default="CGATTAAAGATAGAAATACACG", help="Seed sequence for generation.")
    parser.add_argument("--length_to_generate", type=int, default=100, help="Length of each generated sequence.")
    parser.add_argument("--nb_sequences", type=int, default=100, help="Number of sequences to generate.")
    parser.add_argument("--ref_sequences", type=str, default="", help="File path to the reference sequences.")

    args = parser.parse_args()
    main(args.embed_dim, args.num_heads, args.hidden_dim, args.num_layers, args.lr, 
         args.epochs, args.batch_size, args.seed_sequence, args.length_to_generate, 
         args.nb_sequences, args.ref_sequences)


