"""import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

"""
import torch
from torch.utils.data import Dataset, DataLoader
from utils import MyTrainDataset

class Trainer:
    def __init__(self,
                 model: torch.nn.module,
                 train_data: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 gpu_id: int,
                 save_every: int
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad() 
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id} Epoch{epoch} | batch_size{b_sz} | Steps:{len(self.train_data)}]")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"Epoch {epoch} | training checkpoint saved at checkpoint.pt")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
        train_set = MyTrainDataset(2048)
        model = torch.nn.Linear(20, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int): 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

def main(device, total_epochs, save_every):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=32)
    trainer = Trainer(model, train_data, optimizer, device, save_every)
    trainer.train(total_epochs)

if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    device = 0
    main(device, total_epochs, save_every)


    

#from collective import scatter2






"""
    Great tutorial on blog https://blog.roboflow.com/collective-communication-distributed-systems-pytorch/
    to learn how to manage processes and ranks, and make the collective communication functions work. Now,
    implement those functions from scratch.
"""

"""

def do_gather(rank, world_size):
    group = dist.new_group(list(range(world_size)))
    tensor = torch.tensor([10], dtype=torch.float32)
    if rank == 0:
        tensor_list = [torch.empty(1) for i in range(world_size)]
        dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    else:
        dist.gather(tensor, gather_list=[], dst=0, group=group)

    if rank == 0:
        print(f"rank[{rank}] data = {tensor_list}")
"""
"""
def init_process(rank: int, size: int, fn, backend="gloo"):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "23335"
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, do_gather))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    

    # lauch multiple processes
    processes = [mp.Process(target=gather, args=(tensor, 0)) for _ in range(world_size)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    if dist.get_rank() == 0:
        print(f"gathered tensor: {gathered_tensor}")
"""


"""
def do_reduce(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    # sending all tensors to rank 0 and sum them
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM, group=group)
    # only rank 0 will have 4
    print(f"rank[{rank}] data = {tensor[0]}")

def do_scatter(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.empty(1)
    # sending all tensors from rank 0 to the others
    if rank == 0:
        tensor_list = [torch.tensor([i + 1], dtype=torch.float32) for i in range(size)] # tensor_list = [tensor(1), tensor(2), tensor(3), tensor(4)]
        dist.scatter(tensor, scatter_list=tensor_list, src=0, group=group)
    else:
        dist.scatter(tensor, scatter_list=[], src=0, group=group)
    print(f"rank[{rank}] data = {tensor[0]}")

def do_gather(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    tensor = torch.tensor([rank], dtype=torch.float32)
    if rank == 0:
        tensor_list = [torch.empty(1) for i in range(size)]
        dist.gather(tensor, gather_list=tensor_list, dst=0, group=group)
    else:
        dist.gather(tensor, gather_list=[], dst=0, group=group)
    if rank == 0:
        print(f"rank[{rank}] data = {tensor_list}")

def do_broadcast(rank: int, size: int):
    group = dist.new_group(list(range(size)))
    if rank == 0:
        tensor = torch.tensor([rank], dtype=torch.float32)
    else:
        tensor = torch.empty(1)
    dist.broadcast(tensor, src=0, group=group)
    # all ranks have tensor([0.]) from rank 0
    print(f"rank[{rank}] data = {tensor}")

if __name__ == "__main__":
    size = 6
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, scatter2))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
"""