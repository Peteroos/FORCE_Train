import torch
import math
import random
import numpy as np
import os
import glob
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.sampler import Sampler

from .augment import AugmentPipe

# =============================================================================
# HU normalization (THE FIX)
# -----------------------------------------------------------------------------
# The original code normalized with `hu / 2000 * 2 - 1` on *standard* HU
# (water = 0), then clamped to [-1, 1]. Because every HU <= 0 maps to <= -1, it
# was clamped onto the -1 floor: the trained prior could NOT represent negative
# HU (fat ~ -100, lung / bowel gas / air down to -1024) and rendered them as
# 0 HU. This is why FORCE looked "washed" in the [-160, 240] soft-tissue window.
#
# We instead map the full diagnostic HU range [HU_MIN, HU_MAX] linearly to
# [-1, 1], so the prior covers the whole range including negative HU.
#
# IMPORTANT: inference MUST use the SAME constants:
#     pix = (hu - HU_MIN) / (HU_MAX - HU_MIN) * 2 - 1      # hu  -> [-1, 1]
#     hu  = (pix + 1) / 2 * (HU_MAX - HU_MIN) + HU_MIN     # pix -> hu
# =============================================================================
HU_MIN, HU_MAX = -1024.0, 3071.0


def hu_to_norm(hu):
    """HU -> [-1, 1] model domain (full diagnostic range; covers negative HU)."""
    x = (hu - HU_MIN) / (HU_MAX - HU_MIN) * 2.0 - 1.0
    return x.clamp(-1.0, 1.0) if torch.is_tensor(x) else np.clip(x, -1.0, 1.0)


def norm_to_hu(x):
    """[-1, 1] model domain -> HU."""
    return (x + 1.0) / 2.0 * (HU_MAX - HU_MIN) + HU_MIN


def augment_defaults():
    
    return dict(
        xflip=1,
        yflip=1,
        rotate_int=1,
        translate_int=1,
        scale=1,
        rotate_frac=0,
        aniso=0,
        translate_frac=0,
        brightness=0,
        contrast=0,
        lumaflip=0,
        hue=0,
        saturation=0,
    )

def random_rot(img1,img2):
    k = np.random.randint(0, 3)
    img1 = np.rot90(img1, k+1).copy()
    img2 = np.rot90(img2, k+1).copy()
    return img1,img2

def random_flip(img1,img2):
    axis = np.random.randint(0, 2)
    img1 = np.flip(img1, axis=axis).copy()
    img2 = np.flip(img2, axis=axis).copy()
    return img1,img2

def RadomGenerator(ndct, ldct):
    if random.random() > 0.5:
        ndct, ldct = random_rot(ndct, ldct)
    if random.random() > 0.5:
        ndct, ldct = random_flip(ndct, ldct)
    return ndct, ldct

class MayoDataset(Dataset):
    def __init__(self, root, split='train', **augment_kwargs):
        # Accept pre-converted .npy/.npz OR raw DICOM (.IMA/.dcm) directly --
        # no separate conversion step needed. DICOM is read + HU-rescaled on the fly.
        files = []
        for pat in ('*.npz', '*.npy', '*.IMA', '*.ima', '*.dcm', '*.DCM'):
            files += glob.glob(os.path.join(root, '**', pat), recursive=True)
        self.list = np.array(sorted(set(files)))
        self.data_len = len(self.list)
        self.split = split
        self.data_len = len(self.list)
        self.transformer = transforms.ToTensor()
        self.augmenter = AugmentPipe(**augment_kwargs)
        self.split = split

    def _load_sample(self, data_path):
        low = data_path.lower()
        if low.endswith('.ima') or low.endswith('.dcm'):
            # Read DICOM directly and convert to HU (slope/intercept), same as
            # convert_dicom_to_npy.py -- avoids the .npy pre-conversion step.
            import pydicom
            ds = pydicom.dcmread(data_path, force=True)
            slope = float(getattr(ds, 'RescaleSlope', 1))
            intercept = float(getattr(ds, 'RescaleIntercept', 0))
            hu = ds.pixel_array.astype(np.float32) * slope + intercept
            return hu, None
        data = np.load(data_path)
        if data_path.endswith('.npz'):
            ndct = data['ndct'].astype(np.float32)
            ldct = data['ldct'].astype(np.float32) if 'ldct' in data.files else None
            return ndct, ldct
        return data.astype(np.float32), None
        
    def __getitem__(self, index):
        data_path = self.list[index]
        ndct, ldct = self._load_sample(data_path)
        # Map full diagnostic HU range [HU_MIN, HU_MAX] -> [0, 1] (covers
        # negative HU, unlike the original /2000 which floored HU<=0).
        ndct = (self.transformer(ndct) - HU_MIN) / (HU_MAX - HU_MIN)
        if ldct is not None:
            ldct = (self.transformer(ldct) - HU_MIN) / (HU_MAX - HU_MIN)

        if self.split == 'train':
            ndct = ndct[None, ...]
            ndct, = self.augmenter(ndct)
            ndct = ndct.squeeze(0)

        ndct = ndct * 2 - 1
        ndct = ndct.clamp(-1., 1.)
        if ldct is not None:
            ldct = (ldct * 2 - 1).clamp(-1., 1.)

        if self.split == "train":
            return ndct, {}
        
        path = os.path.splitext(os.path.basename(data_path))[0] + ".npy"
        if self.split == "test":
            return (ldct if ldct is not None else ndct), path

    def __len__(self):
        return self.data_len
    
class GeneralDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        pad: pad data by replicating samples
    """

    def __init__(self, dataset, num_replicas=None, rank=None, pad=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.pad = pad
        self.epoch = 0
        if self.pad:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = int(math.ceil((len(self.dataset)-self.rank) * 1.0 / self.num_replicas))
            self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        if self.pad:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

if __name__ == "__main__":
    dataset = MayoDataset("/gpfs/u/scratch/DTIR/DTIRxnjn/data/mayo/fullImage/3mm", split="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # for i, (ndct, ldct) in enumerate(loader):
    #     print(i, ndct.shape, ldct["x_cond"].shape)
    for i, (ldct, path) in enumerate(loader):
        print(i, ldct.shape, path)
