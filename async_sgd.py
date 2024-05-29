# import random
# import torch
# import torch.distributed as dist
# import torch.distributed.autograd as dist_autograd
# import torch.distributed.rpc as rpc
# import torch.multiprocessing as mp
# import torch.optim as optim
# from torch.distributed.nn import RemoteModule
# from torch.distributed.optim import DistributedOptimizer
# from torch.distributed.rpc import RRef, TensorPipeRpcBackendOptions
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.nn.functional as F

# NUM_EMBEDDINGS = 100
# EMBEDDING_DIM = 16

# class HybridModel(torch.nn.Module):
#     def __init__(self, remote_emb_module, device):
#         super(HybridModel, self).__init__()
#         self.remote_emb_module = remote_emb_module
#         self.fc = torch.nn.Linear(16, 8).to(device)
#         self.device = device

#     def forward(self, indices, offsets):
#         emb_lookup = self.remote_emb_module.forward(indices, offsets).to(self.device)
#         return self.fc(emb_lookup)

# def _run_trainer(remote_emb_module, rank):
#     model = HybridModel(remote_emb_module, rank)
#     model_parameter_rrefs = [RRef(param) for param in model.fc.parameters()]

#     opt = optim.SGD(model.fc.parameters(), lr=0.05)

#     criterion = torch.nn.CrossEntropyLoss().to(rank)

#     def get_next_batch(rank):
#         for _ in range(10):
#             num_indices = random.randint(20, 50)
#             indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS).to(rank)

#             offsets = []
#             start = 0
#             batch_size = 0
#             while start < num_indices:
#                 offsets.append(start)
#                 start += random.randint(1, 10)
#                 batch_size += 1

#             offsets_tensor = torch.LongTensor(offsets).to(rank)
#             target = torch.LongTensor(batch_size).random_(8).to(rank)
#             yield indices, offsets_tensor, target

#     for epoch in range(100):
#         for indices, offsets, target in get_next_batch(rank):
#             output = model(indices, offsets)
#             loss = criterion(output, target)
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
#         print(f"Training done for epoch {epoch} on worker {rank}")

# def run_worker(rank, world_size):
#     rpc_backend_options = TensorPipeRpcBackendOptions()
#     rpc_backend_options.init_method = "tcp://localhost:29501"

#     # Setup device mapping for GPU tensors
#     if rank == 2:
#         # Master node configuration
#         rpc_backend_options.set_device_map('ps', {0: 0})  # Assuming the PS also has a GPU

#     elif rank == 0 or rank == 1:
#         # Trainer nodes configuration
#         rpc_backend_options.set_device_map('ps', {rank: 0})  # Map trainer's GPU to PS's GPU

#     if rank == 2:
#         rpc.init_rpc(
#             "master",
#             rank=rank,
#             world_size=world_size,
#             rpc_backend_options=rpc_backend_options,
#         )
#         # Remaining setup for master...
#     elif rank <= 1:
#         rpc.init_rpc(
#             f"trainer{rank}",
#             rank=rank,
#             world_size=world_size,
#             rpc_backend_options=rpc_backend_options,
#         )
#         # Remaining setup for trainers...
#     else:
#         # Parameter server setup
#         rpc.init_rpc(
#             "ps",
#             rank=rank,
#             world_size=world_size,
#             rpc_backend_options=rpc_backend_options,
#         )
#         # PS does nothing but wait for RPC calls

#     rpc.shutdown()


# if __name__ == "__main__":
#     world_size = 4
#     mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)

import random
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef, TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16

class HybridModel(torch.nn.Module):
    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = torch.nn.Linear(EMBEDDING_DIM, 8).to(device)
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets).to(self.device)
        return self.fc(emb_lookup)

def _run_trainer(remote_emb_module, rank):
    model = HybridModel(remote_emb_module, rank)
    opt = optim.SGD(model.fc.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss().to(rank)
    
    # DataLoader setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Simplified MNIST normalization
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train for 10 epochs (reduced for demo)
    for epoch in range(10):
        total_loss = 0
        for data, target in train_loader:
            indices = torch.randint(0, NUM_EMBEDDINGS, (data.size(0),))
            offsets = torch.arange(data.size(0))
            
            # Forward pass
            output = model(indices, offsets)
            loss = criterion(output, target)
            
            # Backward and optimize
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
        
        # Print training loss
        print(f'Epoch {epoch}, Rank {rank}, Training Loss: {total_loss / len(train_loader)}')
        
        # Evaluate on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                indices = torch.randint(0, NUM_EMBEDDINGS, (data.size(0),))
                offsets = torch.arange(data.size(0))
                
                output = model(indices, offsets)
                _, predicted = torch.max(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy}%')

def run_worker(rank, world_size):
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = f"tcp://localhost:2950{rank+1}"
    rpc_backend_options.set_device_map('ps', {0: 0})  # Assuming PS has a GPU

    if rank == 2:
        rpc.init_rpc("master", rank=rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
        remote_emb_module = RemoteModule("ps", torch.nn.EmbeddingBag, args=(NUM_EMBEDDINGS, EMBEDDING_DIM), kwargs={"mode": "sum"})
        _run_trainer(remote_emb_module, rank)
    elif rank == 0 or rank == 1:
        rpc.init_rpc(f"trainer{rank}", rank=rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size, rpc_backend_options=rpc_backend_options)
    
    rpc.shutdown()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
