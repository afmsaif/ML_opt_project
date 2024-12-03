import pandas as pd
import numpy as np
import copy
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
import tqdm

class Unsup_Dataset:
    def __init__(self, path):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'
        self.df = pd.read_csv(self.path + 'segments.csv')
        self.NFFF = 200  # Limit the number of frequency features or frames (optional)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        # Get the segment ID and data as before
        sid = self.df.iloc[item]['segment_id']
        data = sio.loadmat(self.path + '{}'.format(sid))['data']
        
        # Compute the spectrogram of the data
        _, _, data = signal.spectrogram(data[0, :], fs=5000, nperseg=256, noverlap=128, nfft=1024)

        # Apply the frequency feature length limit and z-score normalization
        data = data[:self.NFFF, :]
        data = stats.zscore(data, axis=1)
        
        # Add an extra dimension for batch compatibility
        data = np.expand_dims(data, axis=0)

        # Select another segment to form seg2 (to use for contrastive loss)
        # For unsupervised, we can sample a different segment from the dataset
        seg2_item = random.choice(range(len(self)))  # Randomly choose another item
        sid2 = self.df.iloc[seg2_item]['segment_id']
        data2 = sio.loadmat(self.path + '{}'.format(sid2))['data']
        _, _, data2 = signal.spectrogram(data2[0, :], fs=5000, nperseg=256, noverlap=128, nfft=1024)
        data2 = data2[:self.NFFF, :]
        data2 = stats.zscore(data2, axis=1)
        data2 = np.expand_dims(data2, axis=0)

        # Simulate the swapped condition (you can randomly swap)
        swapped = np.random.choice([True, False])  # Randomly decide if swapped
        
        # If swapped, we could apply some modification to the segments, if necessary (e.g., time-shifting)
        if swapped:
            # Swap the data (or could apply a noise/augmentation if desired)
            data, data2 = data2, data

        # For unsupervised training, return the two segments and the swapped flag
        return data, data2, swapped

    def split_reviewer(self, reviewer_id):
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        # Splitting based on reviewer_id, as in the original code
        idx = self.df['reviewer_id'] != reviewer_id

        train.df = train.df[idx].reset_index(drop=True)
        valid.df = valid.df[np.logical_not(idx)].reset_index(drop=True)
        return train, valid

    def split_random(self, N_valid):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        train.df = train.df.iloc[N_valid:].reset_index(drop=True)
        valid.df = valid.df.iloc[:N_valid].reset_index(drop=True)
        return train, valid

    def integrity_check(self):
        # Perform an integrity check on the dataset
        try:
            for i in tqdm.tqdm(range(len(self))):
                x = self.__getitem__(i)
        except Exception as exc:
            raise exc

    def remove_powerline_noise_class(self):
        # Optionally remove the "powerline noise" class if needed
        self.df = self.df[self.df['category_id'] != 0]
        self.df['category_id'] = self.df['category_id'] - 1
        self.df = self.df.reset_index(drop=True)
        return self

class Sup_Dataset:
    def __init__(self,path):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'
        self.df = pd.read_csv(self.path + 'segments.csv')
        self.NFFF = 200

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        sid = self.df.iloc[item]['segment_id']
        target = self.df.iloc[item]['category_id']
        data = sio.loadmat(self.path+'{}'.format(sid))['data']
        _,_, data = signal.spectrogram(data[0,:],fs=5000,nperseg=256,noverlap=128,nfft=1024)

        data = data[:self.NFFF,:]
        data = stats.zscore(data,axis=1)
        data = np.expand_dims(data,axis=0)
        return data,target

    def split_reviewer(self,reviewer_id):
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        idx = self.df['reviewer_id']!=reviewer_id

        train.df = train.df[idx].reset_index(drop=True)
        valid.df = valid.df[np.logical_not(idx)].reset_index(drop=True)
        return train,valid

    def split_random(self,N_valid):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        train = copy.deepcopy(self)
        valid = copy.deepcopy(self)

        train.df = train.df.iloc[N_valid:].reset_index(drop=True)
        valid.df = valid.df.iloc[:N_valid].reset_index(drop=True)
        return train,valid

    def integrity_check(self):
        # iterate through dataset and check if all the files might be correctly loaded
        try:
            for i in tqdm.tqdm(range(len(self))):
                x = self.__getitem__(i)
        except Exception as exc:
            raise exc

    def remove_powerline_noise_class(self):
        self.df = self.df[self.df['category_id']!=0]
        self.df['category_id'] = self.df['category_id'] - 1
        self.df = self.df.reset_index(drop=True)
        return self



if __name__ == "__main__":
    
    dataset_mayo = Dataset('/media/chenlab2/hdd51/saif/eplap/DATASET_MAYO/').integrity_check()
