import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer
from models.model import BasicModel
from models.semantic_metadata import load_catalog, load_node_text_embeddings
from dataloader import BasicDataset
import world

class LightGCN_Intent(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN_Intent, self).__init__()
        self.config = config
        self.dataset = dataset
        self.semantic_tag_vocab = []
        self.tag_to_intent_idx = {}
        self.api_tag_indices = None
        self.mashup_tag_indices = None
        self.api_tag_intent_attn = None
        self.api_semantic_intent_centers = None
        # Semantic intent attention selected during the current forward pass.
        # NCA and the semantic injection gate reuse these active intent nodes.
        self._cached_user_intent_attn = None
        self._cached_item_intent_attn = None
        # The representation before node-text fusion.  NCA is applied to this
        # intent-graph representation; the result is then used as the first
        # input to the independent semantic fusion gate.
        self._last_intent_user_embeddings = None
        self._last_intent_item_embeddings = None
        self.sbert_model_name = config.get('sbert_model_name', 'all-MiniLM-L6-v2')
        self.__init_weight()
        
        _, self.api_tag_indices, self.semantic_tag_vocab = self._load_api_semantic_features()
        self.semantic_intent_num = len(self.semantic_tag_vocab)
        self.tag_to_intent_idx = {tag: idx for idx, tag in enumerate(self.semantic_tag_vocab)}
        mashup_tags, _ = load_catalog(
            self.config.get('tag_data_path', world.DATA_PATH), 'mashup', self.num_users
        )
        self.mashup_tag_indices = [
            [self.tag_to_intent_idx[tag] for tag in tags if tag in self.tag_to_intent_idx]
            for tags in mashup_tags
        ]
        unmatched = sum(not indices for indices in self.mashup_tag_indices)
        print(f'[Semantic] fixed Mashup tag connections; {unmatched} Mashups use Gumbel Top-M fallback')

        self.top_m_mashup = int(config['topm_mashup'])
        self.warmup_epochs = config['warmup_epochs']
        self.current_epoch = 0
        self.pretrained_mode = False
        
        self.intent_weight = config.get('intent_weight_fixed', 0.05)
        self.intent_weight_max = config.get('intent_weight_max', 0.2)

        self.user_intent_scale = nn.Parameter(torch.tensor(1.0))
        self.item_intent_scale = nn.Parameter(torch.tensor(1.0))

        self.intent_embedding = nn.Parameter(
            torch.empty(self.semantic_intent_num, self.latent_dim)
        )
        nn.init.normal_(self.intent_embedding, mean=0.0, std=0.01)
        self.intent_projection = nn.Linear(
            self.latent_dim, self.latent_dim, bias=False
        )
        try:
            nn.init.eye_(self.intent_projection.weight)
        except Exception:
            nn.init.normal_(self.intent_projection.weight, std=0.01)

        self.intent_anchor = None

        # Semantic-only, node-adaptive intent injection gates.  Mashups and
        # APIs use independent scalar gates so each node controls how much of
        # its semantic-intent residual is injected into the LightGCN state.
        intent_gate_input_dim = 2 * self.latent_dim
        intent_gate_bias = -1.94591  # logit(0.125), matching the prior semantic gate start

        def make_intent_injection_gate():
            module = nn.Sequential(
                nn.Linear(intent_gate_input_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(module[2].weight)
            nn.init.constant_(module[2].bias, intent_gate_bias)
            return module

        self.mashup_intent_gate = make_intent_injection_gate()
        self.api_intent_gate = make_intent_injection_gate()
        self._cached_mashup_intent_gate = None
        self._cached_api_intent_gate = None
        
        self.intent_freeze_epochs = self.config.get('intent_freeze_epochs', 5)
        self.intent_frozen_until_epoch = self.warmup_epochs + self.intent_freeze_epochs
        print(f"Intent embedding frozen until epoch {self.intent_frozen_until_epoch}")
        self.intent_embedding.requires_grad = False

        self.intent_initialized = False
        
        hidden_dim = self.latent_dim
        proj_dim = self.config.get('contrastive_dim', self.latent_dim)
        self.proj_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.nca_delay = self.config.get('nca_delay', 50)

        # Node-level semantics are intentionally a separate branch from the
        # semantic-intent branch above.  The raw SBERT features are fixed;
        # only the alignment MLP and the semantic fusion gate are trained.
        self.use_node_semantic = bool(self.config.get('use_node_semantic', True))
        self.use_intent_graph = os.environ.get('DVIA_ABLATE_INTENT_GRAPH') != '1'
        if not self.use_intent_graph:
            print('[Ablation] semantic intent graph disabled')
        self._init_node_semantic_branch()
        

        print(f"IntentGCN is ready to go (warmup: {self.warmup_epochs} epochs)")

    def _init_node_semantic_branch(self):
        """Build the independent node-text -> aligned-text -> gate branch."""
        if not self.use_node_semantic:
            self.register_buffer('mashup_node_text_raw', torch.empty(0), persistent=False)
            self.register_buffer('api_node_text_raw', torch.empty(0), persistent=False)
            self.node_semantic_mlp = None
            self.mashup_semantic_gate = None
            self.api_semantic_gate = None
            print('[NodeSemantic] disabled by configuration')
            return

        raw_features = load_node_text_embeddings(
            self.config.get('tag_data_path', world.DATA_PATH),
            self.num_users,
            self.num_items,
            self.sbert_model_name,
            str(self.config.get('device', world.device)),
        )
        raw_features = torch.as_tensor(raw_features, dtype=torch.float32)
        raw_dim = int(raw_features.shape[1])
        self.register_buffer(
            'mashup_node_text_raw', raw_features[:self.num_users].contiguous()
        )
        self.register_buffer(
            'api_node_text_raw', raw_features[self.num_users:].contiguous()
        )

        hidden_dim = max(self.latent_dim, min(2 * self.latent_dim, 512))
        self.node_semantic_mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

        # Independent semantic gates fuse Mashup/API intent and text
        # embeddings after semantic-intent propagation.
        gate_input_dim = 2 * self.latent_dim
        def make_gate_module():
            module = nn.Sequential(
                nn.Linear(gate_input_dim, self.latent_dim),
                nn.Sigmoid(),
            )
            # g multiplies the intent representation.  A positive initial
            # bias therefore starts conservatively from the existing method.
            nn.init.zeros_(module[0].weight)
            nn.init.constant_(
                module[0].bias,
                float(self.config.get('semantic_gate_init_bias', 2.0))
            )
            return module

        self.mashup_semantic_gate = make_gate_module()
        self.api_semantic_gate = make_gate_module()
        self._cached_mashup_semantic_gate = None
        self._cached_api_semantic_gate = None
        print(
            '[NodeSemantic] complete Mashup/API tag+description SBERT features '
            f'loaded ({raw_dim} -> {self.latent_dim}); independent '
            'Mashup/API semantic gates enabled'
        )

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
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            print('user NORMAL distribution initilizer')
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print("[Init] item embeddings randomly initialized; LightGCN learns user/item embeddings from behavior graph")
            
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

    def _split_tags(self, value):
        if pd.isna(value):
            return []
        return [tag.strip().lower() for tag in str(value).split('###') if tag.strip()]

    def _primary_tag(self, value):
        tags = self._split_tags(value)
        return tags[0] if tags else '__unknown__'

    def _load_api_semantic_features(self):
        if self.api_semantic_intent_centers is not None and self.api_tag_indices is not None and self.semantic_tag_vocab:
            return None, self.api_tag_indices, self.semantic_tag_vocab

        data_path = self.config.get('tag_data_path', getattr(world, 'DATA_PATH', None))
        api_file = os.path.join(data_path, 'apisData.xlsx')
        if not os.path.exists(api_file):
            raise FileNotFoundError(f"Tag-based semantic intent requires {api_file}")

        # API rows in apisData.xlsx are not guaranteed to be ordered by the
        # graph's api_id.  Always align metadata through API_desc.csv so the
        # semantic intent view is attached to the same API node as LightGCN.
        aligned_tags, aligned_descriptions = load_catalog(
            data_path, 'api', self.num_items
        )
        api_tags = [list(tags) if tags else ['__unknown__']
                    for tags in aligned_tags]
        descriptions = list(aligned_descriptions)

        vocab = sorted({tag for tags in api_tags for tag in tags})
        tag_to_idx = {tag: idx for idx, tag in enumerate(vocab)}
        api_tag_indices = [[tag_to_idx[tag] for tag in tags] for tags in api_tags[:self.num_items]]

        text_corpus = []
        pair_tag_idx = []
        for api_idx, tags in enumerate(api_tags[:self.num_items]):
            for tag in tags:
                text_corpus.append(f"{tag} {descriptions[api_idx]}".strip())
                pair_tag_idx.append(tag_to_idx[tag])

        sbert = SentenceTransformer(self.sbert_model_name, device=str(world.device))
        sbert_features = sbert.encode(
            text_corpus,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        if sbert_features.shape[1] != self.latent_dim:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(2024)
            projection = torch.randn(
                sbert_features.shape[1],
                self.latent_dim,
                generator=generator,
                dtype=torch.float32
            ) / np.sqrt(max(sbert_features.shape[1], 1))
            pair_features = torch.tensor(sbert_features, dtype=torch.float32) @ projection
        else:
            pair_features = torch.tensor(sbert_features, dtype=torch.float32)

        pair_features = F.normalize(pair_features, p=2, dim=1)
        pair_features[torch.isnan(pair_features)] = 0

        intent_centers = torch.zeros((len(vocab), self.latent_dim), dtype=torch.float32)
        tag_counts = torch.zeros((len(vocab), 1), dtype=torch.float32)

        for row, tag_idx in enumerate(pair_tag_idx):
            intent_centers[tag_idx] += pair_features[row]
            tag_counts[tag_idx] += 1.0

        intent_centers = intent_centers / tag_counts.clamp_min(1.0)
        intent_centers = F.normalize(intent_centers, p=2, dim=1)
        intent_centers[torch.isnan(intent_centers)] = 0

        self.api_semantic_intent_centers = intent_centers
        self.api_tag_indices = api_tag_indices
        self.semantic_tag_vocab = vocab
        return None, self.api_tag_indices, self.semantic_tag_vocab

    def _build_tag_intent_centers(self):
        _, _, _ = self._load_api_semantic_features()
        centers = self.api_semantic_intent_centers
        if centers is None or centers.numel() == 0:
            raise ValueError("Could not build any API categories+description intent prototypes")

        target_num = self.semantic_intent_num
        if centers.shape[0] < target_num:
            repeat = target_num - centers.shape[0]
            centers = torch.cat([centers, centers[:repeat]], dim=0)

        return F.normalize(centers[:target_num], p=2, dim=1)

    def _fixed_api_tag_attention(self, items_emb, intent_emb, scale_param):
        if self.api_tag_indices is None:
            _, self.api_tag_indices, self.semantic_tag_vocab = self._load_api_semantic_features()

        device = items_emb.device
        sqrt_d = np.sqrt(items_emb.size(1))
        rows = []
        cols = []
        vals = []
        for api_idx, tag_indices in enumerate(self.api_tag_indices):
            if not tag_indices:
                continue
            tag_tensor = torch.tensor(tag_indices, dtype=torch.long, device=device)
            scores = torch.matmul(intent_emb.index_select(0, tag_tensor), items_emb[api_idx])
            scores = scores * scale_param / sqrt_d
            weights = F.softmax(scores, dim=0)
            for tag_idx in tag_indices:
                rows.append(api_idx)
                cols.append(tag_idx)
            vals.append(weights)

        row = torch.tensor(rows, dtype=torch.long, device=device)
        col = torch.tensor(cols, dtype=torch.long, device=device)
        val = torch.cat(vals).to(dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=val,
            size=(self.num_items, self.semantic_intent_num),
            device=device
        ).coalesce()

    def _fixed_mashup_tag_attention(self, users_emb, intent_emb, scale_param):
        rows, cols, vals = [], [], []
        unmatched = []
        for user_idx, tag_indices in enumerate(self.mashup_tag_indices):
            if not tag_indices:
                unmatched.append(user_idx)
                continue
            indices = torch.tensor(tag_indices, dtype=torch.long, device=users_emb.device)
            scores = intent_emb.index_select(0, indices) @ users_emb[user_idx]
            weights = F.softmax(scores * scale_param / np.sqrt(users_emb.size(1)), dim=0)
            rows.extend([user_idx] * len(tag_indices))
            cols.extend(tag_indices)
            vals.append(weights)
        indices = torch.tensor([rows, cols], dtype=torch.long, device=users_emb.device)
        values = torch.cat(vals) if vals else users_emb.new_empty(0)
        if unmatched:
            # Map the fallback sub-batch rows back to the full Mashup matrix.
            fallback_rows = torch.tensor(unmatched, dtype=torch.long, device=users_emb.device)
            fallback = self._compute_attention(
                users_emb.index_select(0, fallback_rows), intent_emb,
                scale_param, self.top_m_mashup,
            ).coalesce()
            fallback_indices = torch.stack((
                fallback_rows[fallback.indices()[0]], fallback.indices()[1],
            ))
            indices = torch.cat((indices, fallback_indices), dim=1)
            values = torch.cat((values, fallback.values()))
        return torch.sparse_coo_tensor(
            indices, values, (self.num_users, self.semantic_intent_num),
            device=users_emb.device,
        ).coalesce()

    def init_intent_nodes(self):
        if not self.use_intent_graph:
            self.intent_initialized = False
            print('[Ablation] skip semantic intent prototype initialization')
            return
        print("Initializing semantic intent nodes...")
        centers = self._build_tag_intent_centers().to(world.device)
        noise = torch.randn_like(centers) * 0.01
        self.intent_embedding.data = centers + noise
        self.intent_anchor = centers.detach().cpu().clone()
        print("[Init] tag-driven semantic intents built from all API Categories+Description prototypes")

        if os.environ.get('DVIA_ABLATE_ADAPTER_PRETRAIN') != '1':
            self.adapter_pretrain()
        self.intent_initialized = True
        print("Semantic intent nodes initialized successfully")

    def adapter_pretrain(self):
        for param in self.embedding_user.parameters():
            param.requires_grad = False
        for param in self.embedding_item.parameters():
            param.requires_grad = False

        num_batches = 100
        batch_size = 1024
        optimizer = torch.optim.Adam(self.intent_projection.parameters(), lr=0.001)
        for i in range(num_batches):
            users = torch.randint(0, self.num_users, (batch_size,), device=world.device)
            items = torch.randint(0, self.num_items, (batch_size,), device=world.device)
            with torch.no_grad():
                users_emb = self.embedding_user(users)
                items_emb = self.embedding_item(items)
                original_scores = torch.sum(users_emb * items_emb, dim=1)

            intent_emb = self.intent_projection(self.intent_embedding)
            user_intent_scores = torch.matmul(users_emb, intent_emb.t())
            item_intent_scores = torch.matmul(items_emb, intent_emb.t())
            user_attention = F.softmax(user_intent_scores, dim=1)
            item_attention = F.softmax(item_intent_scores, dim=1)
            user_intent_emb = torch.matmul(user_attention, intent_emb)
            item_intent_emb = torch.matmul(item_attention, intent_emb)
            users_enhanced = users_emb + 0.1 * user_intent_emb
            items_enhanced = items_emb + 0.1 * item_intent_emb
            enhanced_scores = torch.sum(users_enhanced * items_enhanced, dim=1)
            loss = F.mse_loss(enhanced_scores, original_scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for param in self.embedding_user.parameters():
            param.requires_grad = True
        for param in self.embedding_item.parameters():
            param.requires_grad = True
        print("Semantic projection layer pretraining completed")
    
    def update_intent_weight(self):
        if not self.use_intent_graph:
            return
        if not (self.intent_initialized and self.current_epoch >= self.warmup_epochs):
            return

        schedule = self.config.get('intent_schedule', 'fixed')
        if schedule == 'fixed':
            target = self.config.get('intent_weight_fixed', self.intent_weight)
            if abs(self.intent_weight - target) > 1e-9:
                self.intent_weight = target
        elif schedule == 'sigmoid':
            t = float(self.current_epoch - self.warmup_epochs)
            scale_epoch = float(self.config.get('intent_sigmoid_scale', 20.0))
            max_w = float(self.config.get('intent_weight_max', self.intent_weight_max))
            s = 1.0 / (1.0 + torch.exp(- (t/scale_epoch - 4.0)))
            new_w = (s * max_w).item()
            self.intent_weight = max(0.0, min(self.intent_weight_max, new_w))

        if (self.current_epoch >= self.intent_frozen_until_epoch) and (not self.intent_embedding.requires_grad):
            self.intent_embedding.requires_grad = True
            print(f"Unfroze semantic intent_embedding at epoch {self.current_epoch}")
    
    @staticmethod
    def gumbel_softmax_topk(logits: torch.Tensor, k: int, tau: float = 1.0, eps=1e-10):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        gumbel_logits = (logits + gumbel_noise) / tau
        topk_val, topk_idx = torch.topk(gumbel_logits, k, dim=1)
        topk_weight = F.softmax(topk_val, dim=1)
        row = torch.arange(logits.size(0), device=logits.device).unsqueeze(1).expand(-1, k).flatten()
        col = topk_idx.flatten()
        val = topk_weight.flatten()
        sparse_weight = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=val,
            size=logits.shape,
            device=logits.device).coalesce()
        return sparse_weight
    
    def _compute_attention(self, node_emb, intent_emb, scale_param, top_k):
        scores = torch.matmul(node_emb, intent_emb.t())
        sqrt_d = np.sqrt(node_emb.size(1))
        scores = scores * scale_param / sqrt_d
        return self.gumbel_softmax_topk(scores, k=min(top_k, intent_emb.size(0)), tau=1.0)

    def _sparse_colsum(self, mat: torch.sparse.FloatTensor):
        ones = torch.ones((mat.size(0), 1), device=mat.device)
        colsum = torch.sparse.mm(mat.t(), ones)  # [M,1]
        return colsum.squeeze(1)  # [M]

    def _compute_intent_enhanced(self, users_emb, items_emb):
        """Propagate the semantic view and add its intent-aware residuals."""
        if not self.use_intent_graph:
            self._cached_user_intent_attn = None
            self._cached_item_intent_attn = None
            self._cached_mashup_intent_gate = None
            self._cached_api_intent_gate = None
            return users_emb, items_emb
        if self.config.get('normalize_node_emb', False):
            users_emb = F.normalize(users_emb, dim=1)
            items_emb = F.normalize(items_emb, dim=1)

        intent_emb = self.intent_projection(self.intent_embedding)
        user_intent_attn = self._fixed_mashup_tag_attention(
            users_emb, intent_emb, self.user_intent_scale
        )
        item_intent_attn = self._fixed_api_tag_attention(
            items_emb, intent_emb, self.item_intent_scale
        )

        deg_u = self._sparse_colsum(user_intent_attn).clamp_min(1e-6).unsqueeze(1)
        deg_i = self._sparse_colsum(item_intent_attn).clamp_min(1e-6).unsqueeze(1)
        intent_from_user = torch.sparse.mm(user_intent_attn.t(), users_emb) / deg_u
        intent_from_item = torch.sparse.mm(item_intent_attn.t(), items_emb) / deg_i
        intent_emb_updated = (intent_from_user + intent_from_item) / 2

        user_from_intent = (
            torch.sparse.mm(user_intent_attn, intent_emb_updated) * self.intent_weight
        )
        item_from_intent = (
            torch.sparse.mm(item_intent_attn, intent_emb_updated) * self.intent_weight
        )
        if os.environ.get('DVIA_ABLATE_INTENT_GATE') == '1':
            mashup_intent_gate = users_emb.new_full((users_emb.size(0), 1), 0.5)
            api_intent_gate = items_emb.new_full((items_emb.size(0), 1), 0.5)
        else:
            mashup_intent_gate = self.mashup_intent_gate(
                torch.cat([users_emb, user_from_intent], dim=1)
            )
            api_intent_gate = self.api_intent_gate(
                torch.cat([items_emb, item_from_intent], dim=1)
            )
        users_enhanced = users_emb + mashup_intent_gate * user_from_intent
        items_enhanced = items_emb + api_intent_gate * item_from_intent

        if self.training and self.config.get('intent_update_via_ema', False):
            with torch.no_grad():
                self.intent_embedding.mul_(0.9).add_(
                    0.1 * intent_emb_updated.to(self.intent_embedding.device)
                )

        self._cached_user_intent_attn = user_intent_attn
        self._cached_item_intent_attn = item_intent_attn
        self._cached_mashup_intent_gate = mashup_intent_gate
        self._cached_api_intent_gate = api_intent_gate
        return users_enhanced, items_enhanced

    def _compute_node_semantic_embeddings(self, device):
        """Return trainable MLP-aligned complete-text embeddings."""
        if not self.use_node_semantic:
            return None, None
        # SBERT features are deliberately kept out of LightGCN warm-up.  The
        # MLP outputs are normalized here so their scale cannot saturate the
        # independent semantic gate or overwhelm the collaborative branch.
        mashup_text = F.normalize(
            self.node_semantic_mlp(self.mashup_node_text_raw.to(device)),
            p=2, dim=1
        )
        api_text = F.normalize(
            self.node_semantic_mlp(self.api_node_text_raw.to(device)),
            p=2, dim=1
        )
        return mashup_text, api_text

    def _compute_node_semantic_batch(self, raw_features, indices, device):
        """Align text only for sampled nodes during BPR training."""
        selected = raw_features[indices].to(device)
        return F.normalize(self.node_semantic_mlp(selected), p=2, dim=1)

    @staticmethod
    def _fuse_intent_and_semantic(intent_emb, semantic_emb, gate_module):
        gate_input = torch.cat([intent_emb, semantic_emb], dim=1)
        if os.environ.get('DVIA_ABLATE_SEMANTIC_GATE') == '1':
            gate = torch.full_like(intent_emb, 0.5)
        else:
            gate = gate_module(gate_input)
        final_emb = gate * intent_emb + (1.0 - gate) * semantic_emb
        return final_emb, gate

    def _semantic_gated_fusion(self, users_intent_emb, items_intent_emb):
        """Fuse intent/text representations with independent semantic gates."""
        if not self.use_node_semantic:
            return users_intent_emb, items_intent_emb

        mashup_text, api_text = self._compute_node_semantic_embeddings(
            users_intent_emb.device
        )
        users_fused, mashup_gate = self._fuse_intent_and_semantic(
            users_intent_emb, mashup_text,
            self.mashup_semantic_gate,
        )
        items_fused, api_gate = self._fuse_intent_and_semantic(
            items_intent_emb, api_text,
            self.api_semantic_gate,
        )
        self._cached_mashup_semantic_gate = mashup_gate
        self._cached_api_semantic_gate = api_gate
        return users_fused, items_fused

    def computer(self):
        """Return the final recommendation embeddings.

        The order is: collaborative LightGCN -> intent graph propagation
        (the representation constrained by NCA) -> independent node-text
        semantic gated fusion.  During warm-up only the first stage runs.
        """
        if self.pretrained_mode:
            users_emb, items_emb = self._original_lightgcn_forward()
            users_emb, items_emb = self._compute_intent_enhanced(users_emb, items_emb)
            self._last_intent_user_embeddings = users_emb
            self._last_intent_item_embeddings = items_emb
            return self._semantic_gated_fusion(users_emb, items_emb)
        elif self.current_epoch < self.warmup_epochs or not self.intent_initialized:
            # Pure LightGCN warm-up: no intent nodes and no node-semantic
            # branch are active in this phase.
            users_emb, items_emb = self._original_lightgcn_forward()
            self._last_intent_user_embeddings = users_emb
            self._last_intent_item_embeddings = items_emb
            return users_emb, items_emb
        else:
            users_emb, items_emb = self._original_lightgcn_forward()
            users_emb, items_emb = self._compute_intent_enhanced(users_emb, items_emb)
            self._last_intent_user_embeddings = users_emb
            self._last_intent_item_embeddings = items_emb
            return self._semantic_gated_fusion(users_emb, items_emb)

    def _original_lightgcn_forward(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        if self.config['dropout'] and self.training:
            g_droped = self.__dropout(self.keep_prob)
        else:
            g_droped = self.Graph
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        
        return users, items

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        semantic_active = self.use_node_semantic and (
            self.pretrained_mode
            or (self.intent_initialized and self.current_epoch >= self.warmup_epochs)
        )
        if semantic_active:
            # LightGCN and intent propagation still run on the full graph.
            # The node-text branch is independent per node, so computing it
            # only for sampled rows is mathematically identical for BPR and
            # avoids repeatedly transforming every catalog node per batch.
            all_users_intent, all_items_intent = self._original_lightgcn_forward()
            all_users_intent, all_items_intent = self._compute_intent_enhanced(
                all_users_intent, all_items_intent
            )
            self._last_intent_user_embeddings = all_users_intent
            self._last_intent_item_embeddings = all_items_intent

            users_base = all_users_intent[users]
            item_indices = torch.cat([pos_items, neg_items], dim=0)
            items_base = all_items_intent[item_indices]
            users_text = self._compute_node_semantic_batch(
                self.mashup_node_text_raw, users, users_base.device
            )
            items_text = self._compute_node_semantic_batch(
                self.api_node_text_raw, item_indices, items_base.device
            )
            users_emb, mashup_gate = self._fuse_intent_and_semantic(
                users_base, users_text,
                self.mashup_semantic_gate,
            )
            items_fused, api_gate = self._fuse_intent_and_semantic(
                items_base, items_text,
                self.api_semantic_gate,
            )
            pos_emb, neg_emb = torch.split(
                items_fused, [len(pos_items), len(neg_items)], dim=0
            )
            self._cached_mashup_semantic_gate = mashup_gate
            self._cached_api_semantic_gate = api_gate
        else:
            all_users, all_items = self.computer()
            users_emb = all_users[users]
            pos_emb = all_items[pos_items]
            neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def compute_nca_loss_batch(
        self, users_batch_emb, pos_items_batch_emb, users, pos_items,
        temperature=0.07,
    ):
        nca_weight_ui = float(self.config.get('nca_weight_ui', 1.0))
        nca_weight_ii = float(self.config.get('nca_weight_ii', 1.0))
        tau = float(self.config.get('nca_temperature', temperature))

        device = users_batch_emb.device
        intent_pool = self.intent_projection(self.intent_embedding).to(device)
        K = intent_pool.size(0)

        if self._cached_user_intent_attn is None or self._cached_item_intent_attn is None:
            raise RuntimeError(
                'NCA requires semantic intent attention cached by the preceding '
                'intent-enhanced forward pass.'
            )

        # Reuse each node's currently active semantic intents. NCA performs no
        # behavioral-view selection and no second semantic selection.
        mashup_intent_weights = self._cached_user_intent_attn.to_dense()[
            users.long()
        ].to(device)
        api_intent_weights = self._cached_item_intent_attn.to_dense()[
            pos_items.long()
        ].to(device)
        if mashup_intent_weights.size(1) != K or api_intent_weights.size(1) != K:
            raise RuntimeError(
                'Cached semantic attention and semantic intent pool have '
                'incompatible sizes.'
            )

        proj_users = F.normalize(self.proj_mlp(users_batch_emb), dim=1)  # [B, d_proj]
        proj_pos_items = F.normalize(self.proj_mlp(pos_items_batch_emb), dim=1)
        proj_intents = F.normalize(self.proj_mlp(intent_pool), dim=1)  # [K, d_proj]

        sim_u_all = torch.matmul(proj_users, proj_intents.t()) / tau
        sim_v_all = torch.matmul(proj_pos_items, proj_intents.t()) / tau

        exp_sim_u_all = torch.exp(sim_u_all)  # [B, K]
        denom_u = exp_sim_u_all.sum(dim=1)  # [B]
        num_u = (mashup_intent_weights * exp_sim_u_all).sum(dim=1)  # [B]

        denom_u = denom_u.clamp_min(1e-12)
        L_UI = - torch.log((num_u / denom_u).clamp_min(1e-12)).mean()

        exp_sim_v_all = torch.exp(sim_v_all)  # [B, K]
        denom_v = exp_sim_v_all.sum(dim=1)  # [B]

        num_v = (api_intent_weights * exp_sim_v_all).sum(dim=1)  # [B]
        denom_v = denom_v.clamp_min(1e-12)
        L_AI = - torch.log((num_v / denom_v).clamp_min(1e-12)).mean()

        L_nca = nca_weight_ui * L_UI + nca_weight_ii * L_AI
        return L_nca

    def compute_pre_fusion_nca_loss(self, users, pos_items):
        """Compute NCA before node-text fusion for a dedicated optimizer step."""
        if not self.use_intent_graph:
            return None
        nca_lambda = float(self.config.get('lambda_nca', 0.0))
        nca_active = (
            nca_lambda > 0
            and self.intent_initialized
            and (
                self.pretrained_mode
                or self.current_epoch >= self.warmup_epochs + self.nca_delay
            )
        )
        if not nca_active:
            return None

        all_users, all_items = self._original_lightgcn_forward()
        all_users, all_items = self._compute_intent_enhanced(
            all_users, all_items
        )
        self._last_intent_user_embeddings = all_users
        self._last_intent_item_embeddings = all_items
        return self.compute_nca_loss_batch(
            all_users[users.long()],
            all_items[pos_items.long()],
            users,
            pos_items,
            temperature=float(self.config.get('nca_temperature', 0.07)),
        )

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                          posEmb0.norm(2).pow(2) +
                          negEmb0.norm(2).pow(2))/float(len(users))
        
        # In pretrained_mode: intent is always active regardless of epoch
        _joint_branch_active = (
            self.pretrained_mode
            or (self.intent_initialized and self.current_epoch >= self.warmup_epochs)
        )
        _intent_active = self.use_intent_graph and _joint_branch_active
        
        if _intent_active:
            intent_reg = (1/2)*(
                self.intent_embedding.norm(2).pow(2)
                + self.user_intent_scale.norm(2).pow(2)
                + self.item_intent_scale.norm(2).pow(2)
            )/float(len(users))
            reg_loss = reg_loss + intent_reg
            
            
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        
        bpr_loss_val = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # This is the only recommendation BPR in joint training.  It is
        # computed strictly after semantic fusion from final embeddings.
        total_loss = bpr_loss_val
        
        # Track individual loss components for monitoring
        loss_components = {
            'bpr_raw': bpr_loss_val.detach().item(),
            'nca_raw': 0.0,
        }
        return total_loss, reg_loss, loss_components

    def load_lightgcn_pretrained(self, ckpt_path):
        """Load only user/item embedding weights from a pure-LightGCN checkpoint."""
        sd = torch.load(ckpt_path, map_location='cpu')
        self.embedding_user.weight.data.copy_(sd['embedding_user.weight'].to(self.embedding_user.weight.device))
        self.embedding_item.weight.data.copy_(sd['embedding_item.weight'].to(self.embedding_item.weight.device))
        print(f'[LoadLGN] loaded user/item embeddings from {ckpt_path}')

    def load_adapter_pretrained(self, ckpt_path):
        """Load semantic intent projection + embedding state from a checkpoint."""
        sd = torch.load(ckpt_path, map_location='cpu')
        # Older checkpoints stored semantic parameters at view index 1 after
        # behavioral parameters. Migrate only that semantic branch.
        if 'intent_embedding' not in sd and 'intent_embeddings.1' in sd:
            sd['intent_embedding'] = sd['intent_embeddings.1']
        if (
            'intent_projection.weight' not in sd
            and 'intent_projections.1.weight' in sd
        ):
            sd['intent_projection.weight'] = sd['intent_projections.1.weight']
        if 'user_intent_scale' not in sd and 'user_intent_scales.1' in sd:
            sd['user_intent_scale'] = sd['user_intent_scales.1']
        if 'item_intent_scale' not in sd and 'item_intent_scales.1' in sd:
            sd['item_intent_scale'] = sd['item_intent_scales.1']
        sd = {
            key: value for key, value in sd.items()
            if not key.startswith((
                'intent_embeddings.', 'intent_projections.',
                'user_intent_scales.', 'item_intent_scales.',
            )) and key != 'intent_gate'
        }
        msg = self.load_state_dict(sd, strict=False)
        print(f'[LoadAdapter] loaded from {ckpt_path}; missing={msg.missing_keys}, unexpected={msg.unexpected_keys}')

    def enable_pretrained_mode(self):
        self.pretrained_mode = True
        print('[SIGCF] pretrained_mode enabled - warmup gating bypassed')

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


