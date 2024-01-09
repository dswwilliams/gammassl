import torch
import torch.nn as nn
import torch.nn.functional as F


def sharpen(p, dim=1, temp=0.25):
    sharp_p = p**(1./temp)
    sharp_p /= torch.sum(sharp_p, dim=dim, keepdim=True)
    return sharp_p


class TrueCrossEntropy(nn.Module):
    def __init__(self, dim, reduction="mean", class_weights=None, void_class_id=19, gamma_class_weight=1):
        super(TrueCrossEntropy, self).__init__()
        self.dim = dim
        self.reduction = reduction


        self.class_weights = torch.ones(void_class_id+1)
        self.class_weights[void_class_id] = gamma_class_weight
        self.class_weights = self.class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        print("TrueCrossEntropy class weights", self.class_weights.flatten())

    def forward(self, target_probs, query_probs):
        """ 
        xent = sum( - p * log(q) ) = sum*(log(q**-p))
        p: target
        q: input
        """
        
        p = target_probs
        q = query_probs

        p = p + 1e-7
        q = q + 1e-7

        xent = torch.log(q**-p)
        if self.class_weights is not None:
            xent = xent * self.class_weights[:,:xent.shape[1],:,:].to(xent.device)
            
        xent = torch.sum(xent, dim=self.dim, keepdim=False)
        if self.reduction == "mean":
            return xent.mean()
        elif self.reduction == "sum":
            return xent.sum()
        elif self.reduction == "none":
            return xent

def prototype_sep_loss_fn(prototypes, output_metrics=False):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    if output_metrics:
        with torch.no_grad():
            sep = (product - torch.diag(torch.diag(product)) - 1).max()
    # Remove diagonal from loss.
    product -=  2*torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]

    if output_metrics:
        return loss.mean(), sep
    else:
        return loss.mean()


def uniformity_loss_fn(x, t=2, no_exp=False):
    if no_exp:
        return T.pdist(x, p=2).pow(2).mul(-t).mean()
    else:
        return T.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

