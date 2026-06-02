from types import MethodType

import torch


def register_forward(model, model_name):
    """
    Register custom forward method for timm models.

    Note: Modified models (ChannelExpansionWrapper) already have the correct
    forward signature and don't need this registration.
    """
    # Check if model already has the correct forward signature
    # (e.g., ChannelExpansionWrapper from modified_models.py)
    if hasattr(model, 'forward_intermediates') and hasattr(model, 'forward_head'):
        # Check if it's already a modified model
        if model.__class__.__name__ == 'ChannelExpansionWrapper':
            print(f"Model {model_name} is a ChannelExpansionWrapper, skipping forward registration")
            return

    model.forward = MethodType(general_forward, model)

def general_forward(self, x, indices=None, require_feat: bool = True):
    if require_feat:
        x, block_outs = self.forward_intermediates(x, indices)
        x = self.forward_head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



