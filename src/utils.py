import torch.nn.functional as F

# multi-task learning loss function
def MTLLoss(bin_yhat, bin_y, lead_yhat, lead_y, weight=0.6):
    return weight * F.binary_cross_entropy(bin_yhat, bin_y) + (1-weight) * F.nll_loss(lead_yhat, lead_y)

def log_and_print(message, file):
    print(message)
    file.write(message+'\n')