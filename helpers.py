import torch

# note: r means "rotate counterclockwise", f means "flip about vertical axis"

def rb(index=torch.arange(16)): return index[torch.tensor([3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12])]
def fb(index=torch.arange(16)): return index[torch.tensor([3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12])]

def ra(index=torch.arange(4)): return index[torch.tensor([1, 2, 3, 0])]
def fa(index=torch.arange(4)): return index[torch.tensor([0, 3, 2, 1])]

