import random
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP

import random
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define constants
NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16
BATCH_SIZE = 32

# Data transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split dataset into training and testing
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class HybridModel(torch.nn.Module):
    r"""
    The model consists of a sparse part and a dense part.
    1) The dense part is an nn.Linear module that is replicated across all trainers using DistributedDataParallel.
    2) The sparse part is a Remote Module that holds an nn.EmbeddingBag on the parameter server.
    This remote model can get a Remote Reference to the embedding table on the parameter server.
    """

    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(EMBEDDING_DIM, 10).cuda(device), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup.cuda(self.device))

def get_next_batch(dataloader, rank):
    r"""
    Generator function to yield batches from a dataloader.
    """
    for batch in dataloader:
        images, labels = batch
        num_indices = images.size(0)
        indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

        # Use a fixed offset to maintain consistent batch size
        offsets = torch.arange(0, num_indices).long()

        yield indices, offsets, labels.cuda(rank)


def _run_trainer(remote_emb_module, rank):
    r"""
    Each trainer runs a forward pass which involves an embedding lookup on the
    parameter server and running nn.Linear locally. During the backward pass,
    DDP is responsible for aggregating the gradients for the dense part
    (nn.Linear) and distributed autograd ensures gradients updates are
    propagated to the parameter server.
    """

    # Setup the model.
    model = HybridModel(remote_emb_module, rank)

    # Retrieve all model parameters as rrefs for DistributedOptimizer.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # Train for 10 epochs
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training phase
        for indices, offsets, target in get_next_batch(train_loader, rank):
            with dist_autograd.context() as context_id:
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Run distributed optimizer
                opt.step(context_id)

                # Accumulate loss
                running_loss += loss.item()

                # Optional: Calculate training accuracy
                _, predicted = torch.max(output.data, 1)
                total_train += target.size(0)
                correct_train += (predicted == target).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        print(f"Epoch [{epoch + 1}/10], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Testing phase
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for indices, offsets, target in get_next_batch(test_loader, rank):
                output = model(indices, offsets)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        test_accuracy = 100 * correct_test / total_test

        print(f"Epoch [{epoch + 1}/10], Test Accuracy: {test_accuracy:.2f}%")


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """

    # We need to use different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
    if rank == 2:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )

        # Run the training loop on trainers.
        futs = []
        for trainer_rank in [0, 1]:
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank)
            )
            futs.append(fut)

        # Wait for all training to finish.
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        # Initialize process group for Distributed DataParallel on trainers.
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # Initialize RPC.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # Trainer just waits for RPCs from master.
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()

if __name__ == "__main__":
    # 2 trainers, 1 parameter server, 1 master.
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
