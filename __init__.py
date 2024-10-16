# __init__.py

__version__ = "0.1.0"
__author__ = "RachidOunit"
__email__ = "rouni001@cs.ucr.edu"  

from .data_processing import extract_kmers, read_fasta_and_generate_kmers
from .model import TransformerSequenceModel
from .training import train_model
from .sequence_generation import KmerMappingHash, generate_sequences_autoregressively

