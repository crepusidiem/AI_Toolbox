import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def get_attention_map(img, model, num_heads=12, get_mask=False):
    """
    Return a numpy array that'll be ready to be plotted to generate \
    the attention map of the last layer

    Args:
        img: An array-like input with dimension (B, C, H, W), where B is the \
            batch size, C is num_channels, H is height, and W is width
        model: A transformer model from timm
        num_heads: An integer representing the number of heads of the model \
            default 12
        get_mask: Whether to mask the attention map, default False

    Returns:
        A numpy array that'll be ready to be plotted to generate \
        the attention map of the last layer


    """

    # Define a list to store attention weights
    attention_weights = []

    def hook_fn(module, input, output):
        qkv = module.qkv(input[0])  # Forward input through qkv
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape and compute attention weights
        q = q.reshape(q.shape[0], q.shape[1], num_heads, -1).permute(0, 2, 1,
                                                                     3)
        k = k.reshape(k.shape[0], k.shape[1], num_heads, -1).permute(0, 2, 1,
                                                                     3)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (
                    q.shape[-1] ** 0.5)
        attn_weights_tensor = torch.softmax(attn_scores, dim=-1)

        # Store the attention weights in the list
        attention_weights.append(attn_weights_tensor)

    for block in model.blocks:
        # Register the hook
        # block = model.model.blocks[0]
        hook = block.attn.register_forward_hook(hook_fn)

        # Forward pass through the model
        with torch.no_grad():
            _ = model(img)

        # Now, attention_weights contains the captured attention weights

        # Don't forget to remove the hook if you no longer need it
        hook.remove()

    att_mat = torch.stack(attention_weights).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n],
                                           joint_attentions[n - 1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    height, width = img.shape[2], img.shape[3]

    if get_mask:
        result = cv2.resize(mask / mask.max(), (width, height))
    else:
        mask = cv2.resize(mask / mask.max(), (width, height))[..., np.newaxis]

        # Ensure the image tensor has 3 dimensions: (H, W, C)
        if img.dim() == 4:  # (1, 1, H, W)
            img_np = img.squeeze(0).squeeze(
                0).cpu().numpy()  # Convert to (H, W)
        elif img.dim() == 3:  # (1, H, W)
            img_np = img.squeeze(0).cpu().numpy()  # Convert to (H, W)
        else:
            raise ValueError("Unexpected img dimension")

        img_np = np.stack([img_np] * 3,
                          axis=-1)  # Convert grayscale to RGB format

        result = (mask * img_np).astype("uint8")

    return result


def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img, cmap="grey")
    _ = ax2.imshow(att_map)
