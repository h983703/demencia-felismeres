import csv
import os

path = 'wavB'
dir_list = os.listdir(path)
print("Files and directories in '", path, "' :")
print("length: ", len(dir_list))

# prints all files
print(dir_list)

sablon_name = 'txt/B_feature_'
kiterjesztes='_label.csv'
cnt = 0


with open('labels.75.txt','r') as f:
    for line in f:
        for word in line.split():
            print()

            name = ("{}{}{}".format(sablon_name, dir_list[cnt].split('.')[0], kiterjesztes))
            #print(name)
            cnt += 1
            #print(name)
            with open(name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(word)