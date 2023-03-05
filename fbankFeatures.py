import csv
import os
import numpy as np
import python_speech_features as spfeatures
import scipy.io.wavfile as wavfile
# korrelaciohoz
from scipy.stats import pearsonr

basefolder = 'C:/Users/Letti/PycharmProjects/pythonProject/'

path = 'wavB'
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("length: ", len(dir_list))

# prints all files
print(dir_list)

# to append all vectors to one file, e.g. for BoAW
fid = open(basefolder + 'txt/fbank/allFbank.txt', 'w')
# add header for boaw
fid.write('filename')
for i in range(0, 123):  # 123 fbank with energy + delta and delta-delta
    fid.write(',fbank{0:03d}'.format(i + 1))
fid.write('\n')

korrelacio = open(basefolder + 'txt/fbank/probaKorrelacio.txt', 'w')
vektor = open(basefolder + 'txt/fbank/probaFeature.txt', 'w')
sablon_name = 'CSV/B_feature_'
kiterjesztes = '.csv'
cnt = 0
header = ['Feature', 'label']
lepeskoz = [3, 5, 10, 20]

for lepes in lepeskoz:
    cnt = 0
    for z in dir_list:
        name = ("{}{}{}{}".format(sablon_name, dir_list[cnt].split('.')[0], lepes, kiterjesztes))
        print(name)
        korradatok = []
        wavname = z
        print(z)
        # print(path + wavname)
        (rate, wavdata) = wavfile.read(path + '/' + wavname)
        # print('rate',rate)
        # print('wavdata',wavdata)
        wavdata = wavdata / 32768

        fbank_feat = spfeatures.logfbank(wavdata, samplerate=rate, nfilt=40)
        full_feat = fbank_feat

        #   append to huge file
        for i in range(0, full_feat.shape[0]):
            fid.write(wavname)
            # far from optimal but works
            for j in range(0, full_feat.shape[1]):
                fid.write(',{0:.10f}'.format(full_feat[i][j]))
            fid.write('\n')

        # korrelacio elvegzese fbank elvegzese utan

        alen = full_feat.shape[0]
        m = full_feat.shape[1]
        count = 0
        for i in range(0, m):
            for j in range(0, m):
                # temp: valószínűség érték, hogy mennyire biztos benne, ezt tárolja el. az nem kell
                pears_corr, temp = pearsonr(full_feat[:alen - lepes][i], full_feat[lepes:alen][j])
                # print(" korrelacio: ", pears_corr)
                korrelacio.write(',{0:.10f}'.format(pears_corr))
                arr = np.array(pears_corr)
                for x in np.nditer(arr):
                    # print(x)
                    vektor.write(',{0:.10f}'.format(x))
                    # print(type(x))
                    count = count + 1
                    korradatok.append(x)
            korrelacio.write('\n')
            vektor.write('\n')

        A = np.squeeze(np.asarray(korradatok))
        # print(A.shape)
        B = np.reshape(A, (1600, 1))
        # print(B.shape)
        B.tofile(name, sep=',')
        cnt = cnt + 1
        print(cnt)

fid.close()
korrelacio.close()
vektor.close()
