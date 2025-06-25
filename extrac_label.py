import random

with open("E:/Kztech/dataset/plate_rec/train/label_end.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)
n = len(lines)
train = lines[:int(0.8*n)]
val = lines[int(0.8*n):int(0.9*n)]
test = lines[int(0.9*n):]

with open("E:/Kztech/dataset/plate_rec/train/train.txt", "w", encoding="utf-8") as f:
    f.writelines(train)
with open("E:/Kztech/dataset/plate_rec/train/val.txt", "w", encoding="utf-8") as f:
    f.writelines(val)
with open("E:/Kztech/dataset/plate_rec/train/test.txt", "w", encoding="utf-8") as f:
    f.writelines(test)
