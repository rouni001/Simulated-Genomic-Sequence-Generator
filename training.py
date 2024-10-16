import torch
from torch.utils.data import DataLoader
from typing import Optional


def train_model(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: torch.nn.Module, epochs: int, device: str, 
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> None:
    """
    Trains the model on the provided dataset.

    Parameters:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for model parameters.
        criterion (nn.Module): Loss function.
        epochs (int): Number of training epochs.
        device (str): Device for computation (CPU or GPU).
        scheduler (Optional[Optimizer]): Learning rate scheduler.

    Returns:
        None
    """
    model.train()
    max_norm = 1.0  

    for epoch in range(epochs):
        train_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.to(device)
            output = model(batch)

            target_seq = batch[:, 1:]
            output = output[:, :-1, :]

            loss = criterion(output.transpose(1, 2), target_seq)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            if scheduler:
                scheduler.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}')
    
    print("Training complete.")

