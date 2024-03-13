import torch
from collective import scatter

t1 = torch.arange(0, 20)
t1 = t1.view(4, 5)
print(t1)

tlist = scatter(t1, dim_split=0, world_size=4)
#tlist = torch.squeeze(tlist[0])
#for t in tlist:
#    tlist = tlist.insert(t, torch.squeeze(t, dim=0))
print(f"tlist length: {len(tlist)} \n {tlist}")
print(f"tlist[0] shape: {tlist[0].shape}")

