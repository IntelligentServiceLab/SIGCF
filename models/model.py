"""
models/model.py — SIGCF base model classes.

Contains:
  - PairWiseModel:   Interface for pairwise (BPR) loss models.
  - BasicModel:      Base nn.Module inherited by all recommendation models.
  - LightGCN:        Standard LightGCN recommender (He et al., SIGIR'20).

References:
  - LightGCN: https://doi.org/10.1145/3397271.3401063
  - SIGCF:    Semantic Intent Graph Collaborative Filtering.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PairWiseModel:
    """Interface for models that support pairwise (BPR) ranking loss."""
    pass


class BasicModel(nn.Module):
    """
    Base class for all graph-based recommendation models in SIGCF.
    Subclasses must implement:
        computer(self)                     -> (user_embeddings, item_embeddings)
        getUsersRating(self, users)        -> rating scores
        getEmbedding(self, users, pos, neg)-> tuple of embeddings
        bpr_loss(self, users, pos, neg)    -> (loss, reg_loss)
        forward(self, users, items)        -> prediction scores
    """
    def __init__(self):
        super(BasicModel, self).__init__()


class LightGCN(BasicModel, PairWiseModel):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.
    He et al., SIGIR 2020.

    Propagates embeddings over the normalized user–item graph, averages
    layer-wise outputs, and optimises with BPR loss + L2 regularisation.
    """
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.config['pretrain'] == 0:
            # Phase 1 is a pure collaborative LightGCN warm-up.  Keep the
            # initial node embeddings independent of tags/descriptions so the
            # checkpoint contains only Mashup-API graph information.
            nn.init.normal_(self.embedding_user.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.embedding_item.weight, mean=0.0, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config['item_emb']))

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

    def computer(self):
        """LightGCN forward: L layers of propagation, mean-pooling of all layers."""
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        """Compute prediction scores for given users over all items."""
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        """Return both final and initial embeddings for BPR loss."""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        """
        Bayesian Personalized Ranking loss with L2 regularisation.
        Returns (loss, reg_loss).
        """
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(
             users.long(), pos.long(), neg.long())

        reg_loss = (1 / 2) * (
            userEmb0.norm(2).pow(2) +
            posEmb0.norm(2).pow(2) +
            negEmb0.norm(2).pow(2)
        ) / float(len(users))

        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        bpr_loss_val = torch.mean(F.softplus(neg_scores - pos_scores))

        return bpr_loss_val, reg_loss

    def forward(self, users, items):
        """Compute prediction scores for given (user, item) pairs."""
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
