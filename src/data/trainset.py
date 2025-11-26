from .base import *
from torchvision.datasets import CocoCaptions


class COCO(Dataset):
    def __init__(self, json_file, root_dir, crop_size=None, resize_size=None, random_crop=False):
        """
        Args:
            json_file (string): Path to the json file with image paths and captions.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        crop_method = transforms.RandomCrop if random_crop else transforms.CenterCrop
        if crop_size is not None:
            self.transform = transforms.Compose([
                ResizeIfSmall(crop_size),
                crop_method(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif resize_size is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                crop_method(resize_size),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Either crop_size or resize_size must be provided.")
        
        self.coco = CocoCaptions(root=root_dir, annFile=json_file, transform=self.transform)


    def __len__(self):
        return len(self.coco)


    def __getitem__(self, idx):
        img, captions = self.coco[idx]
        img = img * 2 - 1
        caption = rd.choice(captions)
        img_name = f"{idx:012d}"
        
        return {
            'image': img, 
            'caption': caption, 
            'name': img_name, 
        }


class COCO_with_SimpleCaption(Dataset):
    def __init__(self, json_file, root_dir, crop_size=None, resize_size=None, random_crop=False):
        """
        Args:
            json_file (string): Path to the json file with image paths and captions.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        with open(json_file, 'r') as f:
            self.image_captions = json.load(f)
            self.image_names = list(self.image_captions.keys())
        self.root_dir = root_dir
        
        crop_method = transforms.RandomCrop if random_crop else transforms.CenterCrop
        if crop_size is not None:
            self.transform = transforms.Compose([
                ResizeIfSmall(crop_size),
                crop_method(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif resize_size is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                crop_method(resize_size),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError("Either crop_size or resize_size must be provided.")


    def __len__(self):
        return len(self.image_captions)


    def __getitem__(self, idx):
        # open image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) * 2 - 1

        # get caption
        caption = rd.choice(self.image_captions[img_name])

        return {
            'image': image, 
            'caption': caption, 
            'name': img_name, 
        }
        
        
        
class ImageOnlyDataset(Dataset):
    def __init__(self, paths, size=None, random_crop=False):
        self.image_paths = [os.path.join(paths, fname) for fname in os.listdir(paths) if fname.endswith(('.jpg', '.png'))]
        self.size = size
        self.random_crop = random_crop
        
        if self.size is not None and self.size > 0:
            if self.random_crop:
                self.preprocessor = transforms.Compose([
                    ResizeIfSmall(size),
                    transforms.RandomCrop(size),
                    transforms.ToTensor(),
                ])
            else:
                self.preprocessor = transforms.Compose([
                    ResizeIfSmall(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                ])
        else:
            self.preprocessor = transforms.Compose([
                transforms.ToTensor(),
            ])


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, idx):
        # image part
        img_path = self.image_paths[idx]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        image = Image.open(img_path).convert('RGB')
        image = self.preprocessor(image) * 2 - 1
        
        # dummy caption
        caption = ""
        
        output_dict = {
            'image': image,
            'caption': caption,
            'name': img_name,
        }
        
        return output_dict
 
    
    
class SimpleImageText(Dataset):
    def __init__(self, json_file, root_dir, crop_size, resize_size=None, random_crop=False,
                 image_sufx='jpg'):
        """
        Args:
            json_file (string): Path to the json file with image paths and captions.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        with open(json_file, 'r') as f:
            self.image_captions = json.load(f)
            self.image_names = list(self.image_captions.keys())
        self.root_dir = root_dir
        self.image_sufx = f".{image_sufx}"
        
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
        return len(self.image_captions)


    def __getitem__(self, idx):
        # open image
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name+self.image_sufx)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image) * 2 - 1

        # get caption
        caption = self.image_captions[img_name]

        return {
            'image': image, 
            'caption': caption, 
            'name': img_name, 
        }
        