import torch

class TextTokenizerModule:
    def __init__(self, tokenizer_one, tokenizer_two=None):
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
    
    def __call__(self, caption: list, device):
        # embed caption
        text_input_ids_one = self.tokenizer_one(
            caption,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        if self.tokenizer_two:
            text_input_ids_two = self.tokenizer_two(
                caption,
                padding="max_length",
                max_length=self.tokenizer_two.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
        else:
            text_input_ids_two = torch.ones([1], dtype=torch.int32)

        return { 
            'text_input_ids_one': text_input_ids_one.to(device),
            'text_input_ids_two': text_input_ids_two.to(device)
        }
