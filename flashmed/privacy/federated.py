"""Federated Learning for privacy-preserving medical model training."""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from flashmed.registry import PRIVACY_METHODS


@PRIVACY_METHODS.register("federated")
class FederatedLearner:
    """Federated Learning coordinator for multi-hospital model training.

    Implements Federated Averaging (FedAvg) and variants for training models
    across distributed medical data without sharing raw patient data.

    Args:
        model: The global model to train
        num_clients: Number of participating hospitals/sites
        rounds: Number of federated communication rounds
        local_epochs: Training epochs per client per round
        client_fraction: Fraction of clients sampled each round
        learning_rate: Local training learning rate
        aggregation: Aggregation strategy ("fedavg", "fedprox", "scaffold")
        mu: FedProx proximal term weight
    """

    def __init__(
        self,
        model: nn.Module,
        num_clients: int = 5,
        rounds: int = 50,
        local_epochs: int = 5,
        client_fraction: float = 1.0,
        learning_rate: float = 1e-4,
        aggregation: str = "fedavg",
        mu: float = 0.01,
    ):
        self.global_model = model
        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.client_fraction = client_fraction
        self.learning_rate = learning_rate
        self.aggregation = aggregation
        self.mu = mu

        self.global_state = copy.deepcopy(model.state_dict())
        self.round_history: List[Dict[str, float]] = []

    def partition_data(
        self,
        dataset: Dataset,
        strategy: str = "iid",
        alpha: float = 0.5,
    ) -> List[Subset]:
        """Partition dataset across clients.

        Args:
            dataset: Full training dataset
            strategy: "iid" for uniform, "non_iid" for Dirichlet-based partition
            alpha: Dirichlet concentration (lower = more heterogeneous)

        Returns:
            List of dataset subsets, one per client
        """
        n = len(dataset)
        indices = np.arange(n)

        if strategy == "iid":
            np.random.shuffle(indices)
            splits = np.array_split(indices, self.num_clients)
        elif strategy == "non_iid":
            labels = np.array([dataset[i][1] if isinstance(dataset[i][1], int)
                              else dataset[i][1].argmax().item() for i in range(n)])
            num_classes = len(np.unique(labels))
            splits = [[] for _ in range(self.num_clients)]

            for cls in range(num_classes):
                cls_indices = indices[labels == cls]
                proportions = np.random.dirichlet([alpha] * self.num_clients)
                proportions = (proportions * len(cls_indices)).astype(int)
                proportions[-1] = len(cls_indices) - proportions[:-1].sum()

                offset = 0
                for client_id, count in enumerate(proportions):
                    splits[client_id].extend(cls_indices[offset:offset + count].tolist())
                    offset += count
            splits = [np.array(s) for s in splits]
        else:
            np.random.shuffle(indices)
            splits = np.array_split(indices, self.num_clients)

        return [Subset(dataset, split.tolist()) for split in splits]

    def train(
        self,
        client_datasets: List[Dataset],
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Execute federated training.

        Args:
            client_datasets: List of datasets, one per client
            val_loader: Optional global validation loader
            criterion: Loss function
            device: Training device

        Returns:
            Training history
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        print(f"\n{'='*60}")
        print(f"  Federated Learning — {self.aggregation.upper()}")
        print(f"{'='*60}")
        print(f"  Clients:       {self.num_clients}")
        print(f"  Rounds:        {self.rounds}")
        print(f"  Local epochs:  {self.local_epochs}")
        print(f"  Client frac:   {self.client_fraction}")
        print(f"{'='*60}\n")

        for round_idx in range(self.rounds):
            num_selected = max(1, int(self.num_clients * self.client_fraction))
            selected = np.random.choice(self.num_clients, num_selected, replace=False)

            client_updates = []
            client_sizes = []

            for client_id in selected:
                local_model = copy.deepcopy(self.global_model).to(device)
                local_model.load_state_dict(self.global_state)

                local_state, num_samples = self._train_client(
                    local_model, client_datasets[client_id],
                    criterion, device, client_id,
                )
                client_updates.append(local_state)
                client_sizes.append(num_samples)

            self._aggregate(client_updates, client_sizes)
            self.global_model.load_state_dict(self.global_state)

            round_info = {"round": round_idx + 1, "clients_used": len(selected)}

            if val_loader:
                val_metric = self._evaluate(val_loader, device)
                round_info["val_metric"] = val_metric
                print(f"  Round {round_idx+1:3d}/{self.rounds} | "
                      f"Clients: {len(selected)} | Val: {val_metric:.4f}")
            else:
                print(f"  Round {round_idx+1:3d}/{self.rounds} | Clients: {len(selected)}")

            self.round_history.append(round_info)

        return {"history": self.round_history, "final_state": self.global_state}

    def _train_client(
        self,
        model: nn.Module,
        dataset: Dataset,
        criterion: nn.Module,
        device: torch.device,
        client_id: int,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Train a single client locally."""
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()

        global_params = {k: v.clone() for k, v in self.global_state.items()} if self.aggregation == "fedprox" else None

        for epoch in range(self.local_epochs):
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    images, targets = batch[0].to(device), batch[1].to(device)
                else:
                    images = batch["image"].to(device)
                    targets = batch["labels"].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)

                if self.aggregation == "fedprox" and global_params is not None:
                    prox_loss = 0.0
                    for name, param in model.named_parameters():
                        if name in global_params:
                            prox_loss += ((param - global_params[name].to(device)) ** 2).sum()
                    loss += (self.mu / 2) * prox_loss

                loss.backward()
                optimizer.step()

        return model.state_dict(), len(dataset)

    def _aggregate(self, client_states: List[Dict[str, torch.Tensor]], client_sizes: List[int]):
        """Aggregate client model updates using weighted averaging."""
        total_samples = sum(client_sizes)

        new_state = {}
        for key in self.global_state.keys():
            weighted_sum = torch.zeros_like(self.global_state[key], dtype=torch.float32)
            for state, size in zip(client_states, client_sizes):
                weight = size / total_samples
                weighted_sum += state[key].float() * weight
            new_state[key] = weighted_sum

        self.global_state = new_state

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, device: torch.device) -> float:
        """Evaluate global model."""
        self.global_model.eval()
        self.global_model.to(device)
        correct = 0
        total = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0].to(device), batch[1].to(device)
            else:
                images = batch["image"].to(device)
                targets = batch["labels"].to(device)

            outputs = self.global_model(images)
            if targets.dim() > 1:
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == targets).float().mean(dim=1).sum().item()
            else:
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
            total += len(targets)

        return correct / max(total, 1)
