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
fid = open(basefolder + 'txt/mfcc/allMfcc.txt', 'w')
# add header for boaw
fid.write('filename')
for i in range(0, 39):  # 39 mfcc with energy + delta and delta-delta
    fid.write(',mfcc{0:03d}'.format(i + 1))
fid.write('\n')

korrelacio = open(basefolder + 'txt/mfcc/korelacioMfcc.txt', 'w')
vektor = open(basefolder + 'txt/mfcc/vektorKorrMfcc.txt', 'w')
sablon_name = 'CSVMFCC/B_feature_'
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

        mfcc_feat = spfeatures.mfcc(wavdata, samplerate=rate, numcep=13, nfilt=26)
        # mfcc_d_feat = spfeatures.delta(mfcc_feat, 2)
        # mfcc_dd_feat = spfeatures.delta(mfcc_d_feat, 2)

        # full_feat = np.concatenate((mfcc_feat, mfcc_d_feat, mfcc_dd_feat), axis=1)
        full_feat = mfcc_feat

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
        print(A.shape)
        B = np.reshape(A, (169, 1))
        # print(B.shape)
        B.tofile(name, sep=',')
        cnt = cnt + 1
        print(cnt)

fid.close()
korrelacio.close()
vektor.close()
