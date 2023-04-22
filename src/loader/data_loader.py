import os
import torch
import math
import numpy as np
import threading
import queue as Queue
import mxnet as mx
import numbers
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler

from ..tools.data_utils import get_dist_info


def sync_random_seed(seed=None, device="cuda"):
       
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()
import os
import torch
import math
import random
import cv2
import numpy as np
import threading
import queue as Queue
import mxnet as mx
import numbers
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler

from ..tools.data_utils import get_dist_info


def sync_random_seed(seed=None, device="cuda"):
       
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)

    return random_num.item()


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self



class DistributedSampler(_DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,  # world_size
        rank=None,  # local_rank
        shuffle=True,
        seed=0,
    ):

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices * math.ceil(self.total_size / len(indices)))[
            : self.total_size
        ]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DataLoaderX(DataLoader):
    
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch






class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.aug = DataAug()


        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.aug is not None:
            sample = self.aug(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)




class DataAug:

    def __init__(self, **kwargs):
        self.threshold = 0.5
        self.mode_list = list(range(10))

    def break_img(self, img):
        ratio = random.uniform(0.4, 0.6)
        inverse_ratio = 1/ratio
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        img = cv2.resize(img, None, fx=inverse_ratio, fy=inverse_ratio)
        return img

    def strike_h(self, img):
        ratio = random.uniform(1.05, 1.2)
        img = cv2.resize(img, None, fx=ratio, fy=ratio * random.uniform(1.1, 1.3))
        return img

    def add_noise(self, img):
        h, w, c = img.shape
        noise = np.zeros((h, w, c), dtype=np.uint8)
        cv2.randu(noise, 0, 256)
        # Áp dụng ma trận nhiễu vào ảnh
        noisy_img = cv2.add(img, noise)
        return noisy_img

    def rotate_img(self, img):
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),random.randint(-5, 5),1) # xoay 45 độ
        rotated_img = cv2.warpAffine(img,M,(cols,rows))
        return rotated_img

    def resize_img(self, img):
        h, w = img.shape[:2]
        if random.randint(0, 1):
            ratio = random.uniform(0.5, 0.7)
        else:
            ratio = random.uniform(1.3, 1.5)
        img = cv2.resize(img, None, fx=ratio, fy=ratio * random.uniform(0.9, 1.1))
        return img

    def perspective(self, img):
        h, w = img.shape[:2]
       
        target_img = np.ones((int(h * random.uniform(0.6, 0.8)), w + random.randint(2, 5)), dtype=np.uint8) * random.randint(240, 255)
        
        new_h, new_w = target_img.shape[:2]
        target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
        in_points = [[0, 0], [0, h], [w, h], [w, 0]]
        offset_x = random.randint(5, 13)
        offset_noise = random.randint(2, 3)
        if random.randint(0, 1):
            out_points = [[offset_x, 0], [offset_noise, new_h], [w + offset_noise , new_h], [w + offset_x, 0]]
        else:
            out_points = [[offset_noise, 0], [offset_x, new_h], [w, new_h], [w - offset_x, 0]]
        M = cv2.getPerspectiveTransform(np.float32(in_points), np.float32(out_points))
        out = cv2.warpPerspective(img,M,(new_w, new_h),flags=cv2.INTER_LINEAR)
        return out


    def increase_brightness(self, img):
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=random.randint(30, 70))
        return img

    def decrease_brightness(self, img):
        img = cv2.convertScaleAbs(img, alpha=1.0, beta=-random.randint(30, 70))
        return img

    def __call__(self, img):
        if random.uniform(0, 1.0) < self.threshold:
            return img

        mode_choose = random.sample(self.mode_list, random.randint(1, 2))

        for mode in mode_choose:
            if mode == 0:
                img = self.break_img(img)
            elif mode == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif mode == 3:
                img = self.add_noise(img)
            elif mode == 5:
                img = self.resize_img(img)
            elif mode == 7:
                img = self.increase_brightness(img)
            elif mode == 8:
                img = self.decrease_brightness(img)
            else:
                img = img
        img = cv2.resize(img, (112, 112)) 
        return img


if __name__ == '__main__':
    import time
    import glob
    aug = DataAug()
    for i in range(50):
        for img_path in glob.glob("debugs/images/*.*"):
            img_name = img_path.split('/')[-1]
            img = cv2.imread(img_path)
            out_img  = aug(img)
            cv2.imwrite(os.path.join("debugs/aug", f"{time.time()}.jpg"), out_img)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)

    return random_num.item()


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self



class DistributedSampler(_DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas=None,  # world_size
        rank=None,  # local_rank
        shuffle=True,
        seed=0,
    ):

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices * math.ceil(self.total_size / len(indices)))[
            : self.total_size
        ]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DataLoaderX(DataLoader):
    
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch



class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)