import os
import librosa
from pydub import AudioSegment
import numpy as np
from torch.utils.data import Dataset

# Our modules
from vae import configs


def convert_mp3_to_wav(audio_file_path, delete_mp3=False):
    """This function converts an mp3 file to a wav file"""

    if audio_file_path.split(".", 1)[1] == 'mp3':
        try:
            sound = AudioSegment.from_mp3(audio_file_path)
            audio_wav_file_path = audio_file_path.split(".", 1)[0] + ".wav"
            sound.export(audio_wav_file_path, format="wav")  # convert to wav file
            print(audio_file_path, 'file converted from', audio_file_path.split(".", 1)[1], 'to wav format')

            # delete mp3 file
            if delete_mp3:
                os.remove(audio_file_path)

            return audio_wav_file_path

        except:
            raise ValueError('File cannot be converted to wav')
    else:
        raise ValueError('Inserted file is not an mp3 file')


def compute_input(audio_file_path):
    """This function computes the centroid of an audio file given its path"""

    try:
        y, sr = librosa.load(audio_file_path, sr=None)

    except:
        # Call convert_mp3_to_wav to convert mp3 to wav
        new_wav_path = convert_mp3_to_wav(audio_file_path)
        y, sr = librosa.load(new_wav_path, sr=None)

    # centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    cqt = np.abs(librosa.cqt(y,
                             hop_length=configs.InputsConfig.HOP_LENGTH,
                             fmin=configs.InputsConfig.F_MIN,
                             n_bins=configs.InputsConfig.BINS,
                             bins_per_octave=configs.InputsConfig.BINS_PER_OCTAVE))

    return cqt


def store_inputs(dataset_path):

    """This function computes the centroids of all the audio files in
    dataset_path and stores them in data directory in our module."""

    data_path = './data'

    # Create data directory to store the input arrays
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    for (im_dirpath, im_dirnames, im_filenames) in os.walk(dataset_path):
        for f in im_filenames:  # loop in files
            file_name = f.split(".", 1)[0]
            new_path = os.path.join(data_path, file_name + '.npy')

            if os.path.exists(new_path):  # if file is already in data, skip it
                print(new_path, 'already exists')
                continue
            else:
                try:
                    audio_file_path = os.path.join(im_dirpath, f)  # get the audio file path
                    inputs = compute_input(audio_file_path)  # compute the centroid of the audio file
                    np.save(new_path, inputs)  # stores arrays in data directory

                except:
                    print('Skipping file:', f)
                    continue


class AudioDataset(Dataset):

    def __init__(self, data_path, transforms=None):

        """
        Args
        ----
            data_path : Path to all the array files
            audio_file_path : Path of a single audio file
        """

        self.data_path = data_path
        self.input_data = []
        self.files_path = []

        for (im_dirpath, im_dirnames, im_filenames) in os.walk(self.data_path):
            for f in im_filenames:  # loop in files
                if f.split(".", 1)[1] == 'npy':
                    input_file_path = os.path.join(im_dirpath, f)  # get the audio file path
                    input_file_data = np.load(input_file_path)  # load npy file

                    # append variables to lists
                    self.input_data.append(input_file_data)
                    self.files_path.append(input_file_path)

    def __len__(self):
        """count audio files"""
        return len(self.input_data)

    def __getitem__(self, index):
        """take audio file form list"""

        input_data = self.input_data[index]
        input_data = input_data[np.newaxis, :, :]  # add axis for batch

        audio_file = self.files_path[index]

        data = {
                'file': os.path.split(audio_file)[1].split('.')[0],
                'input': input_data
                }

        return data
