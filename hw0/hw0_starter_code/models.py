"""
Classes defining user and item latent representations in
factorization models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=32,
        layer_sizes=[96, 64],
        sparse=False,
        embedding_sharing=True,
    ):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing
        # Create user latent representation
        self.user_emb_for_fact = ScaledEmbedding(
            num_users, embedding_dim, sparse=sparse
        )
        self.user_bias_for_fact = ZeroEmbedding(num_users, 1, sparse=sparse)

        # Create item latent representation
        self.item_emb_for_fact = ScaledEmbedding(
            num_items, embedding_dim, sparse=sparse
        )
        self.item_bias_for_fact = ZeroEmbedding(num_items, 1, sparse=sparse)

        if embedding_sharing:
            self.item_emb_for_reg = self.item_emb_for_fact
            self.item_bias_for_reg = self.item_bias_for_fact
            self.user_emb_for_reg = self.user_emb_for_fact
            self.user_bias_for_reg = self.user_bias_for_fact
        else:
            self.user_emb_for_reg = ScaledEmbedding(
                num_users, embedding_dim, sparse=sparse
            )
            self.item_emb_for_reg = ScaledEmbedding(
                num_items, embedding_dim, sparse=sparse
            )

        # Create the MLP layers
        self.layers = nn.Sequential(*create_layers(layer_sizes, embedding_dim * 3))

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """

        # Get embeddings for factorization
        # Get the user and item latent representations
        user_latent_for_fact = self.user_emb_for_fact(user_ids)
        item_latent_for_fact = self.item_emb_for_fact(item_ids)
        user_bias_for_fact = self.user_bias_for_fact(user_ids).squeeze()
        item_bias_for_fact = self.item_bias_for_fact(item_ids).squeeze()

        # Get embeddings for regression
        user_latent_for_reg = self.user_emb_for_reg(user_ids)
        item_latent_for_reg = self.item_emb_for_reg(item_ids)

        # Concatenate the user and item representations for regression
        user_item_representation = torch.cat(
            [
                user_latent_for_reg,
                item_latent_for_reg,
                user_latent_for_reg * item_latent_for_reg,
            ],
            dim=1,
        )

        # Compute probabilistic prediction
        predictions = (
            (user_latent_for_fact * item_latent_for_fact).sum(1)
            + user_bias_for_fact
            + item_bias_for_fact
        )

        # Forward pass through the MLP to get score for regression
        score = self.layers(user_item_representation).squeeze()

        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")

        return predictions, score


def create_layers(layer_sizes, embedding_dim):
    """
    Create a list of layers for the MLP.

    Parameters
    ----------

    layer_sizes: list
        List of layer sizes to for the regression network.
    embedding_dim: int
        Dimensionality of the latent representations.

    Returns
    -------

    layers: list
        List of layers for the MLP.
    """

    layers = []
    for i, layer_size in enumerate(layer_sizes):
        layers.append(nn.Linear(embedding_dim, layer_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layer_sizes[-1], 1))

    return layers


if __name__ == "__main__":
    net = MultiTaskNet(10, 10, 32, [96, 64], False, True)
    # Create input data
    user_ids = torch.tensor([0, 1, 2, 3, 4])
    item_ids = torch.tensor([0, 1, 2, 3, 4])
    # Forward pass
    predictions, score = net(user_ids, item_ids)
    print(predictions, score)
