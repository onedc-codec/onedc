from .base import *

from datasets.splits import Split
from datasets.arrow_dataset import ArrowReader, Dataset, DatasetInfo


class modified_ArrowReader(ArrowReader):
    def read(self, name, instructions, split_infos, in_memory, length=None):
        files = self.get_file_instructions(name, instructions, split_infos)
        if not files:
            msg = f'Instruction "{instructions}" corresponds to no data!'
            raise ValueError(msg)
        if length is not None:
            files = files[:length]
        return self.read_files(files=files, original_instructions=instructions, in_memory=in_memory)


class CommonCanvas_HF(Dataset):
    def __init__(self, data_dir, crop_size=None, resize_size=None, random_crop=False, loading_length=None, in_memory=False):
        # define base common canvas dataset
        self.dataset_info = DatasetInfo().from_directory(data_dir)
        self.dataset_name = "commoncatalog-cc-by-nd"
        self.dataset_split = Split.TRAIN
        self.in_memory = in_memory
        
        dataset_kwargs = modified_ArrowReader(data_dir, self.dataset_info).read(
            name=self.dataset_name,
            instructions=self.dataset_split,
            split_infos=self.dataset_info.splits.values(),
            in_memory=in_memory,
            length=loading_length
        )
        self.base_dataset = Dataset(**dataset_kwargs)
        
        # define transforms
        transform_list = [ResizeIfSmall(crop_size)]
        if resize_size is not None:
            assert resize_size >= crop_size
            transform_list.append(transforms.Resize(resize_size))
        
        crop_method = transforms.RandomCrop if random_crop else transforms.CenterCrop
        transform_list.append(crop_method(crop_size))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)


    def __len__(self):
        return len(self.base_dataset)


    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img = self.transform(item['jpg'].convert('RGB')) * 2. - 1.
        caption = item['blip2_caption']
        img_name = item['unickname']
        return {
            'image': img, 
            'caption': caption, 
            'name': img_name, 
        }
        
        
        
class CommonCanvas_HF_crop(Dataset):
    def __init__(self, data_dir, crop_size=None, loading_length=None, in_memory=False):
        # define base common canvas dataset
        self.dataset_info = DatasetInfo().from_directory(data_dir)
        self.dataset_name = "commoncatalog-cc-by-nd"
        self.dataset_split = Split.TRAIN
        self.in_memory = in_memory
        
        dataset_kwargs = modified_ArrowReader(data_dir, self.dataset_info).read(
            name=self.dataset_name,
            instructions=self.dataset_split,
            split_infos=self.dataset_info.splits.values(),
            in_memory=in_memory,
            length=loading_length
        )
        self.base_dataset = Dataset(**dataset_kwargs)
        
        # define transforms
        transform_list = [
            ResizeIfSmall(crop_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        self.transform = transforms.Compose(transform_list)


    def __len__(self):
        return len(self.base_dataset)


    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        img = self.transform(item['jpg'].convert('RGB')) * 2. - 1.
        caption = item['blip2_caption']
        img_name = item['unickname']
        return {
            'image': img, 
            'caption': caption, 
            'name': img_name, 
        }