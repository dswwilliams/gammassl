import torch

def load_model_state(seg_net, checkpoint):
    """
    Loads model weights into encoder and decoder.
    """
    keys = seg_net.backbone.load_state_dict(checkpoint['encoder'], strict=False)

    if len(keys.missing_keys) > 0:
        print(f"Missing keys: {keys.missing_keys}")
    if len(keys.unexpected_keys) > 0:
        print(f"Unexpected keys: {keys.missing_keys}")

    seg_net.decoder.load_state_dict(checkpoint['decoder'])
    if seg_net.seg_head is not None:
        seg_net.seg_head.load_state_dict(checkpoint['seg_head'])
    if seg_net.projection_net is not None:
        seg_net.projection_net.load_state_dict(checkpoint['projection_net'])


def load_checkpoint_if_exists(model, save_path):
    """
    If the save_path exists, load the model state from the checkpoint.
    """
    if save_path:
        try:
            checkpoint = torch.load(save_path, map_location=model.device)
            load_model_state(model, checkpoint)
        except FileNotFoundError:
            print(f"Checkpoint file not found: {save_path}")


def get_encoder_state_dict(model):
    """
    Returns the state dictionary of the encoder.
    Accounts for the use of LoRA.
    """
    if model.seg_net.encoder.lora_rank is not None:
        import loralib as lora
        return lora.lora_state_dict(model.seg_net.encoder)
    else:
        return model.seg_net.encoder.state_dict()
