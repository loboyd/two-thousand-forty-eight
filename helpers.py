import torch

# for transforming the action space
# note: r means "rotate counterclockwise", f means "flip about vertical axis"
def r(index=torch.arange(4)): return index[torch.tensor([1, 2, 3, 0])]
def f(index=torch.arange(4)): return index[torch.tensor([0, 3, 2, 1])]

