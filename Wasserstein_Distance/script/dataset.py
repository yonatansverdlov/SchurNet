import h5py
import scipy
import numpy as np
import ot
import time
import multiprocessing as mp
from tqdm import tqdm, trange
import torch


def load_hdf5_data(filename):
    dataset = None
    labels = None
    with h5py.File(filename, 'r') as h5f:
        dataset = h5f['data'][:]
        labels = h5f['label'][:]
    dataset = dataset - np.expand_dims(np.mean(dataset, axis=0), 0)  # center
    # dist = np.max(np.sqrt(np.sum(dataset ** 2, axis=1)), 0)
    # dataset = dataset / dist  # scale
    print("Data shape:", dataset.shape)
    # organize into dictionary = {label: [sample index 1, sample index 2, .... ]}
    types = np.unique(labels)
    label_dict = {}
    for t in types:
        label_dict[t] = []
    for i in range(len(dataset)):
        label_dict[labels[i][0]].append(i)
    return dataset, labels, label_dict


def build_single_cell_data(M1, M2, n=256, pairs=20):
    Ps = []
    Qs = []
    jobs = []
    pool = mp.Pool(processes=20)
    p = (1 / n) * np.ones(n)
    q = (1 / n) * np.ones(n)
    for i in range(pairs):
        P = np.random.choice(M1, size=n)
        Q = np.random.choice(M2, size=n)
        mat = ot.dist(P, Q, metric='euclidean')
        job = pool.apply_async(ot.emd2, args=(p, q, mat))
        jobs.append(job)

        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append(torch.tensor(Q, dtype=torch.float32))
    for job in jobs:
        job.wait()
    dists = np.array([job.get() for job in jobs])
    return Ps, Qs, dists


def build_comprehensive_sampler(raw_data, label_dict, nmin=10, nmax=20, pairs=10, scale=False, order=1):
    numsets = raw_data.shape[0]
    numpoints = raw_data.shape[1]
    all_labels = list(label_dict.keys())
    # construct p_label
    p_label = []
    for i in all_labels:
        p_label.append(len(label_dict[i]) / numsets)

    Ps = []
    Qs = []
    dists = []
    jobs = []
    pool = mp.Pool(processes=20)
    for i in trange(pairs):
        # pick 2 random classes 

        c1 = np.random.choice(all_labels, p=p_label)
        c2 = np.random.choice(all_labels, p=p_label)

        # choose 2 random indices
        pindex = np.random.choice(label_dict[c1])
        P = raw_data[pindex]
        qindex = np.random.choice(label_dict[c2])
        Q = raw_data[qindex]

        psz = np.random.randint(low=nmin, high=nmax)
        P = P[np.random.randint(low=0, high=numpoints, size=psz)]

        qsz = np.random.randint(low=nmin, high=nmax)
        Q = Q[np.random.randint(low=0, high=numpoints, size=qsz)]

        mat = np.power(ot.dist(P, Q, metric='euclidean'), order)
        p = (1 / psz) * np.ones(psz)
        q = (1 / qsz) * np.ones(qsz)
        job = pool.apply_async(ot.emd2, args=(p, q, mat))
        jobs.append(job)

        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append(torch.tensor(Q, dtype=torch.float32))
    for job in jobs:
        job.wait()
    dists = np.array([job.get() for job in jobs])
    dists = np.power(dists, 1 / order)
    return Ps, Qs, dists


def build_multiple_item_dataset(raw_data, items=[1040, 2047], max_pcd=3000, train=True):
    # randomly sample max_pcd samples from items
    numpoints = raw_data.shape[1]
    ref_pcd = []
    for idx in items:
        ref_pcd.append(raw_data[idx])
    nmin = 10
    nmax = 200
    jobs = []
    pool = mp.Pool(processes=20)
    Ps = []
    Qs = []
    for i in trange(max_pcd):
        # choose ref_P and ref_Q
        P = ref_pcd[np.random.choice([0, 1])]
        Q = ref_pcd[np.random.choice([0, 1])]
        # sample 
        psz = np.random.randint(low=nmin, high=nmax)
        qsz = np.random.randint(low=nmin, high=nmax)

        psc = np.random.uniform(low=-2.0, high=2.0)
        qsc = np.random.uniform(low=-2.0, high=2.0)
        pvec = np.tile(np.random.uniform(low=0.0, high=1.0, size=3), (psz, 1))
        qvec = np.tile(np.random.uniform(low=0.0, high=1.0, size=3), (qsz, 1))

        P = P[np.random.randint(low=0, high=numpoints, size=psz)] + psc * pvec
        Q = Q[np.random.randint(low=0, high=numpoints, size=qsz)] + qsc * qvec
        mat = ot.dist(P, Q, metric='euclidean')
        p = (1 / len(P)) * np.ones(len(P))
        q = (1 / len(Q)) * np.ones(len(Q))
        job = pool.apply_async(ot.emd2, args=(p, q, mat))
        jobs.append(job)
        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append(torch.tensor(Q, dtype=torch.float32))
    for job in tqdm(jobs):
        job.wait()
    dists = [job.get() for job in jobs]
    permutation = np.random.permutation(len(dists))
    Ps = np.array(Ps)[permutation]
    Qs = np.array(Qs)[permutation]
    dists = np.array(dists)[permutation]
    if train:
        save_name = '/data/sam/modelnet/data/train-datasets/item-{idx1}-item-{idx2}-pairs-{pairs}'.format(idx1=items[0],
                                                                                                          idx2=items[1],
                                                                                                          pairs=max_pcd)
        np.savez(save_name, P=Ps, Q=Qs, dists=dists)

    return Ps, Qs, dists


def build_single_item_dataset(raw_data, item_idx=300, pairs=500, train=True):
    numsets = raw_data.shape[0]
    numpoints = raw_data.shape[1]
    dimension = raw_data.shape[2]
    Ps = []
    Qs = []
    dists = []
    nmin = 10
    nmax = 200
    ref_point_cloud = raw_data[item_idx]
    jobs = []
    pool = mp.Pool(processes=20)
    for i in trange(pairs):
        # sample random pair of point sets
        psz = np.random.randint(low=nmin, high=nmax)
        qsz = np.random.randint(low=nmin, high=nmax)
        # Scale each P and Q by some random vector
        psc = np.random.uniform(low=-2.0, high=2.0)
        qsc = np.random.uniform(low=-2.0, high=2.0)
        pvec = np.tile(np.random.uniform(low=0.0, high=1.0, size=3), (psz, 1))
        qvec = np.tile(np.random.uniform(low=0.0, high=1.0, size=3), (qsz, 1))
        P = ref_point_cloud[np.random.randint(low=0, high=numpoints, size=psz)] + psc * pvec
        Q = ref_point_cloud[np.random.randint(low=0, high=numpoints, size=qsz)] + qsc * qvec
        # compute OT distance
        mat = ot.dist(P, Q, metric='euclidean')
        p = (1 / len(P)) * np.ones(len(P))
        q = (1 / len(Q)) * np.ones(len(Q))
        job = pool.apply_async(ot.emd2, args=(p, q, mat))
        jobs.append(job)
        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append(torch.tensor(Q, dtype=torch.float32))
    for job in tqdm(jobs):
        job.wait()
    dists = [job.get() for job in jobs]
    permutation = np.random.permutation(len(dists))
    Ps = np.array(Ps)[permutation]
    Qs = np.array(Qs)[permutation]
    dists = np.array(dists)[permutation]
    if train:
        save_name = '/data/sam/modelnet/data/train-datasets/item-{idx}-pairs-{pairs}'.format(pairs=pairs, idx=item_idx)
        np.savez(save_name, P=Ps, Q=Qs, dists=dists)
    else:
        save_name = '/data/sam/modelnet/data/test-datasets/item-{idx}-pairs-{pairs}'.format(pairs=pairs, idx=item_idx)
        np.savez(save_name, P=Ps, Q=Qs, dists=dists)

    return Ps, Qs, dists


def fixed_point_set(dim=2, num=20, data_type='random', low=0.0, high=1.0):
    if data_type == 'random':
        points = np.random.uniform(low=0.0, high=1.0, size=(num, 2))
        return points
    elif data_type == 'grid':
        coords = []
        for _ in range(dim):
            c = np.linspace(0.0, 1.0, num=num)
            coords.append(c)
        meshgrid = np.array(np.meshgrid(*coords))
        grid = []
        for n in range(num):
            grid.append(meshgrid[:, n].T)
        grid = np.vstack(grid)
        return grid
    elif data_type == 'circle':
        points = []
        for i in range(num):
            angle = np.pi * np.random.uniform(0, 2)
            x = 1.0 * np.cos(angle)
            y = 1.0 * np.sin(angle)
            points.append([x, y])
        return np.array(points)

    else:
        raise NameError('not implemented, choose random, grid, or circle')


def build_dataset(points, nmin=5, nmax=20, pairs=1000, order=1):
    Ps = []
    Qs = []
    nmin = nmin
    ntotal = len(points)
    jobs = []
    pool = mp.Pool(processes=20)

    for i in trange(pairs):
        # randomly sample points from two given point sets
        psz = np.random.randint(low=nmin, high=nmax)
        qsz = np.random.randint(low=nmin, high=nmax)
        P = points[np.random.randint(low=0, high=ntotal, size=psz)]
        Q = points[np.random.randint(low=0, high=ntotal, size=qsz)]

        # compute OT distance
        mat = np.power(ot.dist(P, Q, metric='euclidean'), order)
        p = (1 / psz) * np.ones(psz)
        q = (1 / qsz) * np.ones(qsz)
        job = pool.apply_async(ot.emd2, args=(p, q, mat))
        jobs.append(job)
        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append(torch.tensor(Q, dtype=torch.float32))
    for job in tqdm(jobs):
        job.wait()
    dists = np.array([job.get() for job in jobs])
    dists = np.power(dists, 1 / order)
    return Ps, Qs, dists


def noisy_circle_points(sz, radius, dim=2):
    points = []
    for i in range(sz):
        angle = np.pi * np.random.uniform(0, 2)
        r = np.random.normal(loc=radius, scale=0.05)
        if dim == 2:
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
        else:
            vec = np.random.normal(loc=0.0, scale=1.0, size=dim)
            norm = np.linalg.norm(vec, ord=2)
            if norm < 0.0001:
                continue
            vec = radius * (1 / np.linalg.norm(vec, ord=2)) * vec
            points.append(vec)
    return np.array(points)


def noisy_circles(nmin=5, nmax=20, pairs=1000, dim=2, order=1):
    Ps = []
    Qs = []
    nmin = nmin
    jobs = []
    pool = mp.Pool(processes=20)

    radii = [1.0, 0.75, 0.5, 0.25]

    for i in range(pairs):
        # randomly sample points from two given point sets
        psz = np.random.randint(low=nmin, high=nmax)
        qsz = np.random.randint(low=nmin, high=nmax)

        p_radii = np.random.choice(radii)
        q_radii = np.random.choice(radii)
        P = noisy_circle_points(psz, p_radii, dim)
        Q = noisy_circle_points(qsz, q_radii, dim)
        # compute OT distance
        mat = np.power(ot.dist(P, Q, metric='euclidean'), order)

        p = (1 / psz) * np.ones(psz)
        q = (1 / qsz) * np.ones(qsz)
        job = pool.apply_async(ot.emd2, args=(p, q, mat))
        jobs.append(job)
        Ps.append(torch.tensor(P, dtype=torch.float32))
        Qs.append(torch.tensor(Q, dtype=torch.float32))
    for job in jobs:
        job.wait()
    dists = np.power(np.array([job.get() for job in jobs]), 1 / order)
    return Ps, Qs, dists


class PointNetDataloader:
    def __init__(self, Ps, Qs, dists, batch_size, shuffle=False):
        self.Ps = Ps
        self.Qs = Qs
        self.total = len(Ps)
        self.dists = torch.tensor(dists)
        self.shuffle = shuffle
        self.batch_size = batch_size

        # Output tensors for each element
        self.Pblock = torch.cat(self.Ps)
        self.Qblock = torch.cat(self.Qs)
        self.Pidx = []
        pstart = 0
        self.Qidx = []
        qstart = 0
        for i in range(self.total):
            psz = len(self.Ps[i])
            qsz = len(self.Qs[i])
            pend = pstart + psz
            qend = qstart + qsz
            self.Pidx.append([pstart, pend])
            self.Qidx.append([qstart, qend])
            pstart = pend
            qstart = qend
        self.Pidx = torch.tensor(self.Pidx)
        self.Qidx = torch.tensor(self.Qidx)
        self.current_batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        batch_index = self.current_batch * self.batch_size
        if batch_index >= self.total:
            self.current_batch = 0
            raise StopIteration
        Pidx = self.Pidx[batch_index: batch_index + self.batch_size]
        pblock_start = Pidx[0][0]
        pblock_end = Pidx[-1][1]
        Pblock = self.Pblock[pblock_start:pblock_end]

        Qidx = self.Qidx[batch_index:batch_index + self.batch_size]
        qblock_start = Qidx[0][0]
        qblock_end = Qidx[-1][1]
        Qblock = self.Qblock[qblock_start:qblock_end]

        self.current_batch += 1
        return Pblock, Qblock, Pidx, Qidx, self.dists[batch_index:batch_index + self.batch_size]
