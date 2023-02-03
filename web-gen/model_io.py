import torch

# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
def save(path, model, optimizer, other_params={}):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **other_params
    }, path)

def load(path, model, optimizer=None, default_params={}):
    try:
        # https://github.com/pytorch/pytorch/issues/10622
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint
    except Exception as e: 
        print(e)
        return default_params