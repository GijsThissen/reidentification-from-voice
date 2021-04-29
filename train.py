import utils.Data_Loader as DL
import torchaudio
import os
import NN.Siamese_Conv as NN

current_dir = os.getcwd()
data_path = current_dir+"/data/raw_data.pkl"

# preprocessing function to use when loading the data
mfcc = torchaudio.transforms.MFCC(sample_rate =  44100, log_mels=True)

# loading the data
train_data = DL.DataLD(data_path, mfcc)


SN = NN.SiameseNetwork()



for i, data in enumerate(train_data.get_loader(batch_size = 10)):

    print(f"=============\n==============\ntensor: {i}\n============")
    rec0, rec1, lbl = data
    d = SN.forward(rec0, rec1)
    print("euclidian distance between two vectors:", d)
    print()
    print(lbl)
    #print(f"rec0: {rec0}\n \n rec1: {rec1} \n \n lbl: {lbl}")
    if i > 20:
        break
