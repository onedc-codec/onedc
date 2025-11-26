from .base import *
        
class ImageOnlyDataset(Dataset):
    def __init__(self, paths):
        self.image_paths = [os.path.join(paths, fname) for fname in os.listdir(paths) if fname.endswith(('.jpg', '.png'))]
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