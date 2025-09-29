import random

from torch.utils.data import Dataset
from mimiciv_dataset.mimiciv import MIMICIV


class MultipleDataset(Dataset):
    def __init__(self, dataset_list, curriculum=False, seed=42):
        assert len(dataset_list) <= 2
        random.seed(seed)

        self.dataset_list = dataset_list
        self.dataset_len = [len(dataset) for dataset in self.dataset_list]
        self.total_len = sum(self.dataset_len)
        
        if not curriculum:
            total_idx = list(range(self.total_len))
            random.shuffle(total_idx)
        else:
            total_idx = self.get_curriculum_idx()

        self.dataset_idx = []
        self.data_idx = []
        for dataset_idx, dataset_len in enumerate(self.dataset_len):
            self.dataset_idx += [dataset_idx] * dataset_len
            self.data_idx += list(range(dataset_len))
        
        self.dataset_idx = [self.dataset_idx[idx] for idx in total_idx]
        self.data_idx = [self.data_idx[idx] for idx in total_idx]

    def get_curriculum_idx(self):
        total_idx = list(range(self.total_len))
        total_idx = [total_idx[:self.dataset_len[0]], total_idx[self.dataset_len[0]:]]
        random.shuffle(total_idx[0])
        random.shuffle(total_idx[1])

        curriculum_total_idx = []
        dataset_sample_id = [0] * 2
        threshold = (1 - 2 * self.dataset_len[1] / self.dataset_len[0])
        for step in range(self.total_len):
            # p = step / self.total_len
            p = 0.0 if ((step / self.total_len) < threshold) else (((step / self.total_len) - threshold) / (1 - threshold))
            if (random.random() < p and dataset_sample_id[1] < self.dataset_len[1]) or (dataset_sample_id[0] >= self.dataset_len[0]):
                curriculum_total_idx.append(total_idx[1][dataset_sample_id[1]])
                dataset_sample_id[1] += 1
            else:
                curriculum_total_idx.append(total_idx[0][dataset_sample_id[0]])
                dataset_sample_id[0] += 1

        return curriculum_total_idx
    
    def __getitem__(self, idx):
        dataset = self.dataset_list[self.dataset_idx[idx]]

        if isinstance(dataset, MIMICIV):
            sample = dataset[self.data_idx[idx]]
        
        elif isinstance(dataset, list):
            sample = {"messages": dataset[self.data_idx[idx]]}
        
        else:
            raise NotImplementedError

        return sample

    
    def __len__(self):
        return self.total_len


if __name__ == "__main__":
    dataset1 = [f"i_{idx}" for idx in range(1000)]
    dataset2 = [f"j_{idx}" for idx in range(100)]

    dataset = MultipleDataset([dataset1, dataset2], curriculum=True)

    for idx, data in enumerate(dataset):
        print(f"{idx}: {data}")
    
    print(len(dataset))

