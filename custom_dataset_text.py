import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenizer_eng(text):
        return spacy.load('en_core_web_sm')(text.lower())
    
    def build_vocabulary(self, caption_list):
        frequencies = {}
        idx = len(self.stoi)

        for caption in caption_list:
            for token in self.tokenizer_eng(caption):
                if token not in frequencies:
                    frequencies[token] = 1
                else:
                    frequencies[token] += 1

                if frequencies[token] == self.freq_threshold:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1

    def encode_text(self, text):
        tokenized_text = self.tokenizer_eng(text)
        tokenized_text = ['<SOS>'] + tokenized_text + ['<EOS>']

        tokenized = []

        for token in tokenized_text:
            if token in self.stoi.keys():
                tokenized.append(self.stoi[token])
            else:
                tokenized.append(self.stoi['<UNK>'])
            
        return tokenized
    
    def decode_text(self, tokens):
        return ' '.join([self.vocab.itos[token] for token in tokens])
    

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.image = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())


    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.image[index]
        
        img = Image.open(os.path.join(self.root_dir, img_id)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        numericalized_caption += self.vocab.encode_text(caption)
    
        return img, torch.tensor(numericalized_caption)
        

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    

root_dir = 'Flickr8k_Dataset'
captions_file = 'Flickr8k.token.txt'

dataset = Dataset(root_dir, captions_file, transform=None, freq_threshold=5)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=MyCollate(pad_idx=dataset.vocab.stoi['<PAD>']))
