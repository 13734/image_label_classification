import random
import os 
import json

with open("project.json","r",encoding='utf-8') as d:
    str_json = d.read()
    train_json = json.loads(str_json)
tran_test_rate = train_json['train_val']
all_file = r"images_all.txt"
test_file = 'val.txt'
tran_file = 'train.txt'

test_list = []
tran_list = []

    string = f.read()
    list_all = string.split("\n")[:-1]
for i in range(len(list_all)):
    where = random.random() #???
    list_idx = random.randint(0,len(list_all)-1)
    if where < tran_test_rate:
        tran_list.append(list_all[list_idx])
        del(list_all[list_idx])
    else:
        test_list.append(list_all[list_idx])
        del(list_all[list_idx])
    if i%10000 ==0:
        print(i)


with open(tran_file,"w",encoding="utf-8") as d:
    d.write("\n".join(tran_list))

with open(test_file,"w",encoding="utf-8") as d:
    d.write("\n".join(test_list))
