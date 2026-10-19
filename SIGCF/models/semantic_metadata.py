"""ID-aligned catalog text for semantic initialization and Mashup tags."""

from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def split_tags(value):
    if pd.isna(value):
        return []
    return list(dict.fromkeys(t.strip().lower() for t in str(value).split('###') if t.strip()))


@lru_cache(maxsize=8)
def load_catalog(data_path, kind, count):
    root = Path(data_path)
    if kind == 'mashup':
        csv_name, sheet = 'Mashup_desc.csv', 'mashups.xlsx'
        id_col, name_col = 'mashup_id', 'mashup_name'
        raw_name, raw_desc, raw_tags = 'mashups_name', 'description', 'categories'
    elif kind == 'api':
        csv_name, sheet = 'API_desc.csv', 'apisData.xlsx'
        id_col, name_col = 'api_id', 'api_name'
        raw_name, raw_desc, raw_tags = 'APIName', 'Description', 'Categories'
    else:
        raise ValueError(f'Unknown catalog kind: {kind}')
    entities = pd.read_csv(root / csv_name).sort_values(id_col)
    if entities[id_col].tolist() != list(range(count)):
        raise ValueError(f'{csv_name}: expected unique IDs 0..{count - 1}')
    raw = pd.read_excel(root / sheet, usecols=[raw_name, raw_desc, raw_tags])
    groups = {name: rows for name, rows in raw.groupby(raw_name)}
    tags, descriptions = [], []
    missing, merged = [], []
    for _, entity in entities.iterrows():
        name = entity[name_col]
        description = '' if pd.isna(entity['description']) else str(entity['description']).strip()
        descriptions.append(description)
        matches = groups.get(name)
        if matches is None:
            missing.append(name)
            tags.append(())
            continue
        if len(matches) > 1:
            exact = matches[matches[raw_desc].fillna('').str.strip() == description]
            if len(exact):
                matches = exact
            tag_sets = [set(split_tags(value)) for value in matches[raw_tags]]
            if any(value != tag_sets[0] for value in tag_sets[1:]):
                if not len(exact):
                    raise ValueError(f'Ambiguous metadata for {kind} {name!r}')
                merged.append(name)
        tags.append(tuple(dict.fromkeys(
            tag for value in matches[raw_tags] for tag in split_tags(value)
        )))
    if missing:
        warnings.warn(f'{kind}: no tag metadata for {missing}; using description only.')
    if merged:
        warnings.warn(f'{kind}: merged tags of identical-name/description records: {merged}')
    return tuple(tags), tuple(descriptions)


@lru_cache(maxsize=8)
def load_node_text_embeddings(data_path, num_users, num_items, model_name, device='cpu'):
    """Encode each Mashup/API's complete catalog text for node semantics.

    The returned matrix is deliberately kept in the original SBERT feature
    space.  LightGCN never consumes this matrix directly: a trainable MLP in
    ``LightGCN_Intent`` performs the dimension alignment and nonlinear mapping
    after the pure collaborative warm-up.
    """
    from sentence_transformers import SentenceTransformer

    user_tags, user_desc = load_catalog(data_path, 'mashup', num_users)
    item_tags, item_desc = load_catalog(data_path, 'api', num_items)
    texts = [
        f"Categories: {' '.join(tags)}. Description: {description}".strip()
        for tags, description in zip(user_tags + item_tags, user_desc + item_desc)
    ]
    encoder = SentenceTransformer(model_name, device=str(device))
    features = encoder.encode(
        texts,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != num_users + num_items:
        raise ValueError(
            'SBERT node semantic encoder returned an unexpected shape: '
            f'{features.shape}; expected ({num_users + num_items}, d)'
        )
    if not np.isfinite(features).all():
        raise ValueError('Non-finite SBERT node semantic embeddings')
    return features


def initialize_semantic_embeddings(user_embedding, item_embedding, config):
    from sentence_transformers import SentenceTransformer

    root = config['tag_data_path']
    user_tags, user_desc = load_catalog(root, 'mashup', user_embedding.num_embeddings)
    item_tags, item_desc = load_catalog(root, 'api', item_embedding.num_embeddings)
    texts = [f"{' '.join(tags)} {description}".strip()
             for tags, description in zip(user_tags + item_tags, user_desc + item_desc)]
    encoder = SentenceTransformer(
        config.get('sbert_model_name', 'all-MiniLM-L6-v2'),
        device=str(config.get('device', 'cpu')),
    )
    features = torch.as_tensor(encoder.encode(
        texts, batch_size=128, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ), dtype=torch.float32)
    dimension = user_embedding.embedding_dim
    if features.shape[1] != dimension:
        generator = torch.Generator(device='cpu').manual_seed(2024)
        projection = torch.randn(features.shape[1], dimension, generator=generator) / np.sqrt(features.shape[1])
        features = features @ projection
    features = F.normalize(features, p=2, dim=1)
    if not torch.isfinite(features).all():
        raise ValueError('Non-finite semantic initialization embeddings')
    with torch.no_grad():
        user_embedding.weight.copy_(features[:user_embedding.num_embeddings])
        item_embedding.weight.copy_(features[user_embedding.num_embeddings:])
    print('[Init] Mashup and API embeddings initialized from their own tags + descriptions')
