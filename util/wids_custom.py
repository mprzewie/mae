"""incorporate https://github.com/webdataset/webdataset/pull/382/commits/bdb5de09c78abcd59804748bf9dcd63cada5dc8a"""

import base64
import gzip
import hashlib
import io
import math
import os
import random
import re
import sqlite3
import sys
import uuid
import warnings
from functools import partial
from typing import Any, BinaryIO, Dict, Optional, TypeVar, Union
from urllib.parse import quote, urlparse

import numpy as np
import torch.distributed as dist

from torch.utils.data import Dataset, Sampler
from typing import Optional
from wids.wids import iterate_ranges

class ChunkedSampler(Sampler):
    """A sampler that samples in chunks and then shuffles the samples within each chunk.

    This preserves locality of reference while still shuffling the data.
    """

    def __init__(
        self,
        dataset,
        *,
        dslength_per_replica=-1,
        num_samples=None,
        chunksize=2000,
        seed=0,
        shuffle=True,
        shufflefirst=False,
    ):
        if isinstance(num_samples, int):
            lo, hi = 0, num_samples
        elif num_samples is None:
            lo, hi = 0, len(dataset)
        else:
            lo, hi = num_samples

        self.dslength_per_replica = (
            dslength_per_replica if dslength_per_replica > 0 else (hi - lo)
        )
        self.ranges = [(i, min(i + chunksize, hi)) for i in range(lo, hi, chunksize)]
        self.seed = seed
        self.shuffle = shuffle
        self.shufflefirst = shufflefirst
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.dslength_per_replica

    def __iter__(self):
        self.rng = random.Random(self.seed + 1289738273 * self.epoch)
        shardshuffle = self.shufflefirst or self.epoch > 0
        yield from iterate_ranges(
            self.ranges,
            self.rng,
            indexshuffle=self.shuffle,
            shardshuffle=(self.shuffle and shardshuffle),
        )
        self.epoch += 1


def DistributedChunkedSampler(
    dataset: Dataset,
    *,
    num_replicas: Optional[int] = None,
    num_samples: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    shufflefirst: bool = False,
    seed: int = 0,
    drop_last: bool = None,
    chunksize: int = 1000000,
) -> ChunkedSampler:
    """Return a ChunkedSampler for the current worker in distributed training.

    Reverts to a simple ChunkedSampler if not running in distributed mode.

    Since the split among workers takes place before the chunk shuffle,
    workers end up with a fixed set of shards they need to download. The
    more workers, the fewer shards are used by each worker.
    """
    if drop_last is not None:
        warnings.warn(
            "DistributedChunkedSampler does not support drop_last, thus it will be ignored"
        )
    if not dist.is_initialized():
        warnings.warn(
            "DistributedChunkedSampler is called without distributed initialized; assuming single process"
        )
        num_replicas = 1
        rank = 0
    else:
        num_replicas = num_replicas or dist.get_world_size()
        rank = rank or dist.get_rank()
    assert rank >= 0 and rank < num_replicas

        # From https://github.com/pytorch/pytorch/blob/13fa59580e4dd695817ccf2f24922fd211667fc8/torch/utils/data/distributed.py#L93
    dslength_per_replica = (
        math.ceil(len(dataset) / num_replicas) if num_replicas > 1 else len(dataset)
    )
    
    num_samples = num_samples or len(dataset)
    worker_chunk = (num_samples + num_replicas - 1) // num_replicas
    worker_start = rank * worker_chunk
    worker_end = min(worker_start + worker_chunk, num_samples)
    return ChunkedSampler(
        dataset,
        num_samples=(worker_start, worker_end),
        dslength_per_replica=dslength_per_replica,
        chunksize=chunksize,
        seed=seed,
        shuffle=shuffle,
        shufflefirst=shufflefirst,
    )