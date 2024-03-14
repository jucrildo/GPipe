import torch
from collective import scatter, broadcast

t1 = torch.arange(0, 20)
t1 = t1.view(4, 5)
print(t1)

# scatter 
tlist = scatter(t1, dim_split=0, world_size=4)
print(f"tlist length: {len(tlist)} \n {tlist}")
print(f"tlist[0] shape: {tlist[0].shape}")

# broadcast
t3 = torch.arange(0, 4).view(2, 2)
print(f"t3: \n{t3}")
tlist3 = broadcast(t3, world_size=4)
print(f"tlist3.len: {len(tlist3)}\n{tlist3}")
print(f"tlist3[0]: {tlist3[0].shape}")
