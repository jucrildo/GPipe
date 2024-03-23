# helper functions
import torch
from typing import List

def _check_tensors_dtype(tensor_list: List[torch.tensor], tensor_dtype: torch.dtype):
    for t in tensor_list:
        if t.dtype != tensor_dtype:
            return False
    return True

def _check_tensors_size(tensor_list: List[torch.tensor]):
    for t in tensor_list:
        if t.shape != tensor_list[0].shape:
            return False 
    return True

# squeeze each tensor of the list. (find a better way to do that)
def _squeeze_list(tensor_list: List[torch.tensor]) -> List[torch.tensor]:
    list = []
    #if isinstance(tensor_list, List[torch.tensor]):
    for t in tensor_list:
        list.append(torch.squeeze(t)) 
    return list

def _rank_in_group(rank: int, process_group: List[int]) -> bool:
    for proc in process_group:
        if rank == proc:
            return True
        else:
            return False



"""
dim = 0
t1 = torch.arange(0, 20)
n = 4
print(t1.view(4, 5).shape)
print(t1.view(4, 5))

tensor_tuple = []
if t1.size(dim) % n == 0:
    tensor_tuple = (torch.split(t1, n, dim=dim))

print(len(tensor_tuple))
print(tensor_tuple[0].dtype)

t2 = torch.ones(3, 4, 2)
print(f"t2 shape: {t2.shape}, t2 shape len: {len(t2.shape)}")
print(len(t1.shape))
"""
    