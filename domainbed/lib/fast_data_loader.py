# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import os

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is None:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)
        else:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)

        # if weights is None:
        #     weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True
        )
        self._infinite_iterator = iter(self._data_loader)

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

    def close(self):
        if getattr(self, "_infinite_iterator", None) is not None:
            shutdown = getattr(self._infinite_iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            self._infinite_iterator = None
        self._data_loader = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers, random=True):
        super().__init__()

        if random:
            batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(dataset, replacement=False),
                batch_size=batch_size,
                drop_last=False
            )
        else:
            batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(dataset),
                batch_size=batch_size,
                drop_last=False
            )

        num_workers = int(os.environ.get('DATALOADER_NUM_WORKERS', num_workers))
        self._data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True
        )
        self._infinite_iterator = iter(self._data_loader)

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length

    def close(self):
        if getattr(self, "_infinite_iterator", None) is not None:
            shutdown = getattr(self._infinite_iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
            self._infinite_iterator = None
        self._data_loader = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)