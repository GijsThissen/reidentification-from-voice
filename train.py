import utils.Data_Loader as DL
import torchaudio

data_path = "/home/jakob/data/raw_data.pkl"


mfcc = torchaudio.transforms.MFCC(sample_rate =  44100, log_mels=True)


#cnn
train_data = DL.DataLD(data_path, mfcc)

for i, data in enumerate(train_data.get_loader()):
    print(f"=============\n==============\ntensor: {i}\n============")
    rec0, rec1, lbl= data
        #print(f"rec0: {rec0}\n \n rec1: {rec1} \n \n lbl: {lbl}")
    if i > 20:
        break
