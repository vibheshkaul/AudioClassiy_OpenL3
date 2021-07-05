import torch
import torch.nn as nn
import torchaudio.transforms as T
import random


class AudioNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        filt_size = (3, 3)
        pool_size = (2, 2)
        num_classes=50
        # Create melspectrograms 
        self.mel_spectrogram = T.MelSpectrogram(
          sample_rate=48000,
          n_mels=128,
          n_fft = 2048,
          hop_length = 242
          )
        self.use_bias = True
        self.b_norm2d = nn.BatchNorm2d(1)
        self.mlp = nn.Sequential(
            
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,num_classes),
            nn.Softmax(dim=1)

        )

        self.pool = nn.Sequential(
            nn.MaxPool2d((16,24)),
            nn.Flatten()           
            
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            #block1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=filt_size, bias=self.use_bias, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=filt_size, bias=self.use_bias, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size, stride=2),

            #block2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=filt_size, bias=self.use_bias,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=filt_size, bias=self.use_bias, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size, stride=2),

            #block3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=filt_size, bias=self.use_bias, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=filt_size, bias=self.use_bias, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size, stride=2),

            #block4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=filt_size, bias=self.use_bias, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=filt_size, bias=self.use_bias, padding=1),
        )
    @staticmethod
    def get_Random_One_Sec_Audio_Sample(X):
          sample_len = X.shape[1]
          st = random.randint(0,sample_len-48000)
          end = st+48000
          samples = X[:,st:end].clone()
          #print(samples.shape)
          return samples

    @staticmethod
    def get_Audio_Set_For_Test(X):
          X = X.unfold(1,48000,24000)
          return X.swapaxes(0,1)

    def transform_to_melSpectogram(self,aud_batch):
          melspec = self.mel_spectrogram(aud_batch)
          return melspec.unsqueeze(1)
        

    def forward(self, X: torch.Tensor):
      if self.training:
            #X is batch of data sample (batch_size x sample_rate)
            
            #Take 1s sample at randmom
            samples = self.get_Random_One_Sec_Audio_Sample(X) # 32 1s samples
            #print(samples.shape)
            #Mel transform
            inp = self.transform_to_melSpectogram(samples) # batch of 32 1s mel-spectrograms
            #print(inp.shape)
            output = self.conv(inp) #batch of 32*512 vector values
            #print(output.shape)
            output = self.pool(output)
            #print(output.shape)
            #MLP 
            output = self.mlp(output)
            return output
      else:
          # X is a set of batches with repeated classes
          audio_set = self.get_Audio_Set_For_Test(X)
          # Loop over the samples in the batch audio_set.shape[0]
          outputs = torch.empty(9,audio_set.shape[1],50)
          for i,sample_set in enumerate(audio_set):
            #print(sample_set.shape)
            inp = self.transform_to_melSpectogram(sample_set)
            #print(inp.shape)
              #Batch Norm
            inp = self.b_norm2d(inp)
            output = self.conv(inp) #512 values
            output = self.pool(output)
            #MLP 
            output = self.mlp(output)

            outputs[i]=output          
          
          return outputs.mean(dim=0)

